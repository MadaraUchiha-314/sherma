# Plan: Conversation Summarization Node Type

## Overview

Add a built-in `summarize` node type that declarative agents can use to manage context window usage in long-running, multi-turn conversations. The node counts tokens in the active message window, and when a configurable threshold is exceeded, summarizes older messages using an LLM while keeping the most recent N messages intact.

## Design Decisions

### Token Counting Strategy
- Use LangChain's `get_num_tokens_from_messages()` on the bound chat model to count tokens, since the model knows its own tokenizer.
- Fall back to a simple `len(str(msg)) / 4` heuristic if the model doesn't support token counting.

### State Contract
- The node operates on **user-defined state fields** (not hardcoded), following the existing pattern where `summary_field` and `cursor_field` reference state keys.
- Messages are always read from the `messages` state field (standard convention).
- The node returns state updates for `summary_field`, `cursor_field`, and optionally `messages`.

### Re-summarization
- If accumulated summaries themselves exceed the threshold, the node re-summarizes them into a single condensed summary before proceeding.

## Implementation Steps

### Step 1: Define `SummarizeArgs` in `schema.py`

Add a new Pydantic model:

```python
class SummarizeArgs(BaseModel):
    """Arguments for a summarize node."""
    llm: RegistryRef | None = None
    model_context_window: int
    threshold: float = 0.75
    messages_to_keep: int = 6
    summary_field: str = "summary_messages"
    cursor_field: str = "summarized_until"
    prompt: str | None = None  # CEL expression for custom summarization prompt
    exclude_message_types: list[str] = Field(default_factory=list)
```

Update `NodeDef.type` literal to include `"summarize"`.
Update `NodeDef.args` union to include `SummarizeArgs`.
Update the `type_map` in `_resolve_args_type` to include `"summarize": SummarizeArgs`.

### Step 2: Implement `build_summarize_node()` in `nodes.py`

Create the node builder function following the established pattern (see `build_interrupt_node` for reference):

1. **Resolve LLM** — look up from registry via `RegistryRef`, or fall back to `default_llm`.
2. **Count tokens** — compute token count of `state[summary_field] + state.messages[state[cursor_field]:]`.
3. **Check threshold** — if `token_count / model_context_window < threshold`, return early (no-op).
4. **Filter messages** — separate messages into:
   - `to_summarize`: `messages[cursor:len-messages_to_keep]` (excluding types in `exclude_message_types`)
   - `to_keep`: last `messages_to_keep` messages
5. **Summarize** — call the LLM with the existing summaries + messages to summarize, using the configured prompt (or a sensible default).
6. **Re-summarize check** — if the new summary + kept messages still exceed the threshold, re-summarize the accumulated summaries.
7. **Return state updates** — `{summary_field: [new_summary_msg], cursor_field: new_cursor}`.

Hook lifecycle: `node_enter` → (logic) → `node_exit`, plus `before_llm_call` / `after_llm_call` on the summarization LLM invocation.

### Step 3: Register the builder in `agent.py`

In `_build_graph()`, add the `"summarize"` case to the node-type dispatch that calls `build_summarize_node(ctx, cel)`.

### Step 4: Add tests

- **Unit tests** (`tests/test_summarize_node.py`):
  - Token counting triggers summarization when threshold exceeded.
  - No-op when below threshold.
  - `messages_to_keep` preserves the correct recent messages.
  - `cursor_field` advances correctly.
  - `exclude_message_types` filters correctly.
  - Re-summarization of accumulated summaries.
  - Custom prompt CEL expression is evaluated.
  - Fallback to default LLM when `llm` is not specified.

- **Integration test** (`tests/test_summarize_integration.py`):
  - End-to-end YAML agent with a `summarize` node in the graph cycle.

### Step 5: Update docs and skill

- Update `docs/README.md` with the new `summarize` node type documentation.
- Update `skills/sherma/references/` with matching content.
- Update `skills/sherma/SKILL.md` if the quick reference or API surface listing needs changes.

## Files to Modify

| File | Change |
|------|--------|
| `sherma/langgraph/declarative/schema.py` | Add `SummarizeArgs`, update `NodeDef` |
| `sherma/langgraph/declarative/nodes.py` | Add `build_summarize_node()` |
| `sherma/langgraph/declarative/agent.py` | Wire up `"summarize"` in node dispatch |
| `tests/test_summarize_node.py` | New unit tests |
| `tests/test_summarize_integration.py` | New integration test |
| `docs/README.md` | Document `summarize` node type |
| `skills/sherma/references/` | Mirror docs updates |
| `skills/sherma/SKILL.md` | Update quick reference |

## Open Questions

1. **Token counting accuracy** — Should we require a specific tokenizer library (e.g., `tiktoken`) as a dependency, or rely solely on LangChain's built-in counting?
2. **Reducer dependency** — The issue mentions FR-4 (custom reducers) for the `summary_messages` `replace` behavior. If custom reducers aren't implemented yet, the `summary_field` will use the default list append reducer, which means the node must return the full replacement list each time.
