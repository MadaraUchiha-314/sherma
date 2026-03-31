# Plan: call_llm state_updates mapping

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
- Must be backward-compatible: existing agents with no `state_updates` field should behave identically.
- Emit a warning if `state_updates` omits `messages` on a `call_llm` node that has tools bound.
- Should follow existing patterns (`set_state` uses `values: dict[str, str]` with CEL expressions).
- The LLM response should be available as a CEL variable so users can extract parts of it (content, tool_calls, etc.).

---

## 2. Design

### New optional field: `state_updates` on `CallLLMArgs`

Add an optional `state_updates` field to `CallLLMArgs` that works like `set_state.values` -- a dict of state keys to CEL expressions evaluated against the current state **plus** the LLM response.

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
      state_updates:
        summary: 'llm_response.content'
```

### Behavior

| `state_updates` field | Result dict |
|---|---|
| **Not specified** (default) | `{"messages": [response]}` -- current behavior, unchanged |
| **Specified** | Each key-value in `state_updates` is evaluated as a CEL expression. The response is **not** auto-appended to `messages` unless the user explicitly includes `messages` in the output mapping. |

### CEL context for `state_updates` expressions

The CEL evaluation context includes everything in `state` plus:
- `llm_response` -- the raw AIMessage object, exposed as a dict-like with keys:
  - `llm_response.content` (str) -- the text content of the response
  - `llm_response.tool_calls` (list) -- tool calls, if any
  - `llm_response` (object) -- the full response object (for appending to messages: `state.messages + [llm_response]`)

### Examples

**Store content in a custom field (don't append to messages):**
```yaml
state_updates:
  summary: 'llm_response.content'
```

**Append to messages AND store content separately:**
```yaml
state_updates:
  messages: 'state.messages + [llm_response]'
  last_response: 'llm_response.content'
```

**Append to a custom list field:**
```yaml
state_updates:
  history: 'state.history + [llm_response]'
```

---

## 3. Implementation Steps

### Step 1: Schema changes (`sherma/langgraph/declarative/schema.py`)
- Add `state_updates: dict[str, str] | None = None` to `CallLLMArgs`.

### Step 2: Node builder changes (`sherma/langgraph/declarative/nodes.py`)
- In `build_call_llm_node` / `call_llm_fn`, after obtaining the `response`:
  - If `args.state_updates` is `None`: keep current behavior (`{"messages": [response]}`).
  - If `args.state_updates` is set: build a CEL context that includes the current state plus `llm_response` (the AIMessage serialized to a dict-like). Evaluate each key-value pair in `args.state_updates` and construct the result dict from those evaluations.
- The `after_llm_call` hook still fires before output mapping (so it can modify `response`).
- The `node_exit` hook still fires after output mapping (so it can modify the final `result`).

### Step 3: CEL engine support
- Verify that the CEL engine can handle a merged context (state + `llm_response`). May need to pass an augmented activation dict to `cel.evaluate()`.
- Serialize the AIMessage to a dict for CEL consumption (content, tool_calls, etc.).

### Step 4: Tests
- Default behavior (no `state_updates` field) -- verify backward compat.
- `state_updates` mapping to a single custom field.
- `state_updates` mapping to multiple fields.
- `state_updates` that explicitly includes `messages` (append + custom field).
- `state_updates` with `llm_response.content` extraction.
- `state_updates` with `llm_response.tool_calls` extraction.
- Verify `after_llm_call` and `node_exit` hooks still work with output mapping.

### Step 5: Update docs and skill references
- Update `docs/declarative-agents.md` with the `state_updates` field documentation under the `call_llm` section.
- Update `skills/sherma/references/` (copies of docs).
- Update `skills/sherma/SKILL.md` if it covers `call_llm` args.

---

## 4. Resolved Questions

1. **Naming**: `state_updates` — makes it explicit that these are updates to state fields.
2. **Tool call safety**: Emit a **warning** if `state_updates` is set on a `call_llm` node that has tools but doesn't include `messages` in the mapping. The tool execution loop needs the AIMessage in `messages` to work correctly.

---

## Plan Revisions

- **2026-03-31**: Renamed `output` → `state_updates` per owner feedback. Resolved tool call safety question: emit a warning (not error) when `state_updates` omits `messages` on a tooled node.
