# Task 41: Custom State Field Reducers (Replace vs. Append)

**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/41
**Priority:** P0

## Summary

Add an optional `reducer` field to `StateFieldDef` so YAML authors can explicitly choose between `append` and `replace` semantics for list fields. Currently, the `messages` field always uses LangGraph's `MessagesState` append reducer, and all other fields use simple replacement. There's no way to have a list field with replace semantics or a non-messages list field with append semantics.

## Acceptance Criteria

- `StateFieldDef` gains an optional `reducer` field with values: `"append"` (default for `messages`), `"replace"` (default for everything else)
- When `reducer: append` is set on a list field, LangGraph's add-message reducer is used
- When `reducer: replace` is set on a list field, the returned value overwrites the previous value
- The `messages` field continues to default to `append` for backward compatibility
- Non-list fields ignore the `reducer` setting (always replace)
