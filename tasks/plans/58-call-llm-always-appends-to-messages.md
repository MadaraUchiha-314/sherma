# Plan: call_llm output mapping

**Task:** [tasks/58-call-llm-always-appends-to-messages.md](../58-call-llm-always-appends-to-messages.md)
**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/58

---

## 1. Background & Analysis

### Current State
- `call_llm` in `nodes.py:404` always produces `result = {"messages": [response]}`.
- LangGraph's `MessagesState` reducer automatically **appends** this to `state.messages`.
- The `node_exit` hook can modify `result` after the fact, but this requires writing a Python hook for what should be a declarative concern.
- `set_state` already supports writing CEL expressions to arbitrary state keys via `values: dict[str, str]`.

### Problem
Users have no declarative way to:
1. Skip appending the LLM response to `messages`.
2. Store the LLM response content in a different state field (e.g., `state.summary`).
3. Append the response to a custom list field instead of `messages`.

### Design Constraints
- Must be backward-compatible: existing agents with no `output` field should behave identically.
- Should follow existing patterns (`set_state` uses `values: dict[str, str]` with CEL expressions).
- The LLM response should be available as a CEL variable so users can extract parts of it (content, tool_calls, etc.).

---

## 2. Design

### New optional field: `output` on `CallLLMArgs`

Add an optional `output` field to `CallLLMArgs` that works like `set_state.values` -- a dict of state keys to CEL expressions evaluated against the current state **plus** the LLM response.

```yaml
nodes:
  - name: summarizer
    type: call_llm
    args:
      llm: { id: openai-gpt-4o-mini, version: "1.0.0" }
      prompt:
        - role: system
          content: '"Summarize the conversation."'
        - role: messages
          content: 'state.messages'
      output:
        summary: 'llm_response.content'
```

### Behavior

| `output` field | Result dict |
|---|---|
| **Not specified** (default) | `{"messages": [response]}` -- current behavior, unchanged |
| **Specified** | Each key-value in `output` is evaluated as a CEL expression. The response is **not** auto-appended to `messages` unless the user explicitly includes `messages` in the output mapping. |

### CEL context for `output` expressions

The CEL evaluation context includes everything in `state` plus:
- `llm_response` -- the raw AIMessage object, exposed as a dict-like with keys:
  - `llm_response.content` (str) -- the text content of the response
  - `llm_response.tool_calls` (list) -- tool calls, if any
  - `llm_response` (object) -- the full response object (for appending to messages: `state.messages + [llm_response]`)

### Examples

**Store content in a custom field (don't append to messages):**
```yaml
output:
  summary: 'llm_response.content'
```

**Append to messages AND store content separately:**
```yaml
output:
  messages: 'state.messages + [llm_response]'
  last_response: 'llm_response.content'
```

**Append to a custom list field:**
```yaml
output:
  history: 'state.history + [llm_response]'
```

---

## 3. Implementation Steps

### Step 1: Schema changes (`sherma/langgraph/declarative/schema.py`)
- Add `output: dict[str, str] | None = None` to `CallLLMArgs`.

### Step 2: Node builder changes (`sherma/langgraph/declarative/nodes.py`)
- In `build_call_llm_node` / `call_llm_fn`, after obtaining the `response`:
  - If `args.output` is `None`: keep current behavior (`{"messages": [response]}`).
  - If `args.output` is set: build a CEL context that includes the current state plus `llm_response` (the AIMessage serialized to a dict-like). Evaluate each key-value pair in `args.output` and construct the result dict from those evaluations.
- The `after_llm_call` hook still fires before output mapping (so it can modify `response`).
- The `node_exit` hook still fires after output mapping (so it can modify the final `result`).

### Step 3: CEL engine support
- Verify that the CEL engine can handle a merged context (state + `llm_response`). May need to pass an augmented activation dict to `cel.evaluate()`.
- Serialize the AIMessage to a dict for CEL consumption (content, tool_calls, etc.).

### Step 4: Tests
- Default behavior (no `output` field) -- verify backward compat.
- `output` mapping to a single custom field.
- `output` mapping to multiple fields.
- `output` that explicitly includes `messages` (append + custom field).
- `output` with `llm_response.content` extraction.
- `output` with `llm_response.tool_calls` extraction.
- Verify `after_llm_call` and `node_exit` hooks still work with output mapping.

### Step 5: Update docs and skill references
- Update `docs/declarative-agents.md` with the `output` field documentation under the `call_llm` section.
- Update `skills/sherma/references/` (copies of docs).
- Update `skills/sherma/SKILL.md` if it covers `call_llm` args.

---

## 4. Open Questions

1. **Naming**: `output` vs `result` vs `state_updates` -- `output` is concise and consistent with the concept of "what this node outputs to state". Open to alternatives.
2. **Tool call flow**: When `call_llm` has tools and the LLM makes tool calls, the auto-injected `tool_node` still needs the AIMessage in `messages` for the tool execution loop to work. Should we warn or error if `output` is set on a `call_llm` node that has tools but doesn't include `messages` in the output mapping? Or trust the user?

---

## Plan Revisions

_(none yet)_
