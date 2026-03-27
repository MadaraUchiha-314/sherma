# Task 45: Prompt String Templating with State Variables

## Description

Add a `template(str, map)` CEL function that performs `${key}` substitution from a map, making multi-line prompts with multiple injection points more readable than CEL string concatenation.

## Acceptance Criteria

- [x] Add `template(str, map)` CEL function that performs `${key}` substitution from the map
- [x] Unresolved placeholders are left as-is (no error)
- [x] Works with any string, not just prompt references
- [x] Non-string values are coerced to strings
- [x] Unit and integration tests added
- [x] Docs and skill updated

## References

- Issue: https://github.com/MadaraUchiha-314/sherma/issues/45
- Agent Gateway CEL reference reviewed: no existing `template`/`format` function exists in the community standard
