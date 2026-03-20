# Task 28: Accessing state in CEL should be through state prefix

## Description

Currently, state attributes in CEL expressions are accessed directly by their field names (e.g., `messages`, `query_count`). This makes it confusing because there's no distinction between state fields and other CEL variables (like `prompts`, `llms`).

The change requires all state access in CEL to go through a `state` prefix:
- Before: `messages[size(messages) - 1]`
- After: `state.messages[size(state.messages) - 1]` or `state["messages"][size(state["messages"]) - 1]`

Extra variables like `prompts` and `llms` remain at the top level.

## Reference

GitHub Issue: https://github.com/MadaraUchiha-314/sherma/issues/28
