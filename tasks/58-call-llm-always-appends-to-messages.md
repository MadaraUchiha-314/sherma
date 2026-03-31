# Task: call_llm always appends to messages

**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/58

## Problem

- `call_llm` always returns `{"messages": [response]}`, hardcoding the LLM response into the `messages` state field.
- Users cannot control what happens with the LLM response (e.g., store it in a different field, transform it, skip appending to messages).
- Users cannot append the LLM response to arbitrary state attributes beyond `messages`.

## Requirements

1. Users should be able to control what the `call_llm` node does with the LLM response.
2. Users should be able to map the LLM response (or parts of it) to any state field, not just `messages`.
3. The default behavior (appending to `messages`) should be preserved for backward compatibility.

## Chat Iterations

### Iteration 1 (2026-03-31)
- **Naming**: Owner chose `state_updates` over `output` — makes the intent explicit.
- **Tool call safety**: Owner confirmed: emit a warning (not error) when `state_updates` omits `messages` on a `call_llm` node with tools bound.

### Iteration 2 (2026-03-31)
- **Reducer semantics**: Owner flagged that `state_updates` values go through LangGraph's field reducers (e.g., `add_messages` for `messages`). Clarified that values are **deltas**, not final state — consistent with how all LangGraph node returns work. Fixed examples: `'[llm_response]'` (delta) instead of `'state.messages + [llm_response]'` (would duplicate).
