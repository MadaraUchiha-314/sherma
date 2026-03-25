# Task 32: CEL Expressive Power for Structured Data

GitHub Issue: https://github.com/MadaraUchiha-314/sherma/issues/32

## Problem

CEL string parsing is limited — `.contains()` works for keyword routing but cannot parse structured data (JSON) from message content. This limits richness of interrupt-resume protocols.

## Solution

Register custom CEL functions in the cel-python engine, inspired by [agentgateway's CEL reference](https://agentgateway.dev/docs/standalone/latest/reference/cel/).

### Tier 1: JSON Functions
- `json(string) -> map|list` — Parse JSON string into CEL map/list
- `jsonValid(string) -> bool` — Check if string is valid JSON

### Tier 2: Safe Access
- `default(value, fallback) -> value` — Return fallback if value errors (requires special handling since it wraps error-prone expressions)

### Tier 3: String Extensions (aligned with cel-go strings extension)
- `split(string, separator) -> list`
- `trim(string) -> string`
- `lowerAscii(string) -> string`
- `upperAscii(string) -> string`
- `replace(string, old, new) -> string`
- `indexOf(string, substr) -> int`
- `join(list, separator) -> string`
- `substring(string, start, end) -> string`

## Acceptance Criteria

- All three tiers implemented and tested
- Docs and skill references updated with new CEL functions
- All existing tests continue to pass
