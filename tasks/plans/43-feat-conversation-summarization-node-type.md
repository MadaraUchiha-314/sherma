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
- The node returns state updates for `summary_field`, `cursor_field`, `summary_cursor_field`, and optionally `messages`.

### Dual-Cursor Design

The node maintains **two cursors** to handle unbounded growth of both messages and summaries:

1. **`cursor_field`** (message cursor) — tracks how far into the `messages` list we have summarized. Messages before this index have already been condensed into summaries.
2. **`summary_cursor_field`** (summary cursor) — tracks how far into the `summary_messages` list we have re-summarized. Summaries before this index have been condensed into a single meta-summary.

**Flow:**

```
messages:  [m0, m1, m2, ..., m50, m51, ..., m99, m100, ..., m110]
                                    ^cursor                   ^latest
            ── already summarized ──┘── to_summarize ──┘── keep ──┘

summaries: [s0, s1, s2, ..., s20, s21, ..., s30]
                               ^summary_cursor    ^latest
            ── re-summarized ──┘── to_re_summarize ┘
```

**When the node runs:**
1. Check if `summary_messages` tokens exceed the threshold → if yes, re-summarize older summaries (those after `summary_cursor`), advance `summary_cursor`.
2. Check if `summary_messages + unsummarized messages` tokens exceed the threshold → if yes, summarize older messages into a new summary entry, advance `cursor`.

This prevents the summary list from growing without bound in very long conversations.

### Re-summarization
- When accumulated summaries exceed the threshold, the node summarizes summaries from `summary_cursor` to end into a single condensed summary, then advances `summary_cursor`.
- This is a recursive safety net — in practice most conversations will only trigger message-level summarization.

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
    summary_cursor_field: str = "summary_summarized_until"
    prompt: str | None = None  # CEL expression for custom summarization prompt
    exclude_message_types: list[str] = Field(default_factory=list)
```

Update `NodeDef.type` literal to include `"summarize"`.
Update `NodeDef.args` union to include `SummarizeArgs`.
Update the `type_map` in `_resolve_args_type` to include `"summarize": SummarizeArgs`.

### Step 2: Implement `build_summarize_node()` in `nodes.py`

Create the node builder function following the established pattern (see `build_interrupt_node` for reference):

1. **Resolve LLM** — look up from registry via `RegistryRef`, or fall back to `default_llm`.
2. **Phase 1 — Re-summarize summaries (if needed):**
   a. Read `summary_messages` from `state[summary_field]` and `summary_cursor` from `state[summary_cursor_field]`.
   b. Count tokens of `summary_messages[summary_cursor:]`.
   c. If exceeds threshold: call LLM to condense `summary_messages[summary_cursor:]` into a single summary. Append to summary list, advance `summary_cursor` to point past the condensed entries.
3. **Phase 2 — Summarize messages (if needed):**
   a. Read `messages` from state and `cursor` from `state[cursor_field]`.
   b. Count tokens of `summary_messages + messages[cursor:]`.
   c. If below threshold: return early (no-op).
   d. Split messages into `to_summarize = messages[cursor : len-messages_to_keep]` (filtering out `exclude_message_types`) and `to_keep = messages[-messages_to_keep:]`.
   e. Call LLM with existing summaries + `to_summarize`, using the configured prompt (or a sensible default).
   f. Append new summary to `summary_messages`, advance `cursor` to `len(messages) - messages_to_keep`.
4. **Return state updates** — `{summary_field: updated_summaries, cursor_field: new_cursor, summary_cursor_field: new_summary_cursor}`.

Hook lifecycle: `node_enter` → (logic) → `node_exit`, plus `before_llm_call` / `after_llm_call` on each LLM invocation.

### Step 3: Register the builder in `agent.py`

In `_build_graph()`, add the `"summarize"` case to the node-type dispatch that calls `build_summarize_node(ctx, cel)`.

### Step 4: Add tests

- **Unit tests** (`tests/test_summarize_node.py`):
  - Token counting triggers summarization when threshold exceeded.
  - No-op when below threshold.
  - `messages_to_keep` preserves the correct recent messages.
  - `cursor_field` advances correctly.
  - `exclude_message_types` filters correctly.
  - Summary list re-summarization triggers when summaries exceed threshold.
  - `summary_cursor_field` advances correctly after re-summarization.
  - Dual-cursor independence: message cursor and summary cursor advance independently.
  - Custom prompt CEL expression is evaluated.
  - Fallback to default LLM when `llm` is not specified.

- **Integration test** (`tests/test_summarize_integration.py`):
  - End-to-end YAML agent with a `summarize` node in the graph cycle.

### Step 5: Update docs and skill

- Update `docs/README.md` with the new `summarize` node type documentation.
- Update `skills/sherma/references/` with matching content.
- Update `skills/sherma/SKILL.md` if the quick reference or API surface listing needs changes.

## Example YAML Usage

```yaml
state:
  fields:
    - name: messages
      type: list
      default: []
    - name: summary_messages
      type: list
      default: []
      reducer: replace
    - name: summarized_until
      type: int
      default: 0
    - name: summary_summarized_until
      type: int
      default: 0

nodes:
  - name: summarize_if_needed
    type: summarize
    args:
      llm: { id: openai-gpt-4o-mini }
      model_context_window: 128000
      threshold: 0.75
      messages_to_keep: 6
      summary_field: summary_messages
      cursor_field: summarized_until
      summary_cursor_field: summary_summarized_until
      prompt: 'prompts["summarize"]["instructions"]'
      exclude_message_types:
        - loaded_skills
```

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
3. **`model_context_window` removal** — Research shows no framework infers context window size client-side. OpenAI Agents SDK uses item-count heuristics, Claude auto-compaction is server-side, LangGraph uses user-configured `max_token_limit`. The built-in node should drop `model_context_window` and use either a user-configured token budget or message-count threshold instead.

## Reference Implementation (Existing Features)

An example implementation using only existing sherma features lives at
`examples/conversation_summarization/`. It demonstrates the dual-cursor
pattern with `call_llm`, `set_state`, conditional edges, and `interrupt`.

### Limitations of the YAML-only approach (motivating the built-in node type)

1. **Summary pollution** — `call_llm` always appends its response to `messages`, so summary LLM responses appear in the conversation history. A built-in node would write to a dedicated state field instead.
2. **No list slicing in CEL** — Cannot pass only unsummarized messages to the LLM. The summarize `call_llm` receives ALL messages. A built-in node would slice `messages[cursor:len-keep]` in Python.
3. **Message-count thresholds only** — CEL cannot count tokens. A built-in node would use `model.get_num_tokens_from_messages()`.
4. **No message deletion** — Cannot remove already-summarized messages from `messages` to free context. A built-in node could use LangGraph's `RemoveMessage`.

## Plan Revisions

- **Rev 1**: Added dual-cursor design. The original plan had a single `cursor_field` for messages only, with a vague "re-summarize if too big" step. Revised to use two independent cursors (`cursor_field` for messages, `summary_cursor_field` for summaries) so that summary list growth is bounded. Phase 1 (re-summarize summaries) now runs before Phase 2 (summarize messages) to free up space before deciding whether new message summarization is needed.
- **Rev 2**: Dropped `model_context_window` as a node arg based on research (OpenAI, Claude, LangGraph all avoid client-side context window inference). Added reference implementation using existing features at `examples/conversation_summarization/` with documented limitations that motivate the built-in node type.
