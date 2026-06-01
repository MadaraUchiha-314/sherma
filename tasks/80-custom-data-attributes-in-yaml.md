# 80 — Custom data attributes in declarative YAML

GitHub issue: https://github.com/MadaraUchiha-314/sherma/issues/80

## Problem

- sherma's declarative YAML has no section where a user can define arbitrary
  **static data** that can be referenced elsewhere in the YAML via CEL.
- Today CEL expressions can reference a fixed set of "preset" entities
  (`state`, `prompts`, `llms`, `skills`, and the runtime-only `llm_response`),
  but there is no user-defined namespace for constants (thresholds, label maps,
  feature flags, routing tables, etc.).
- Secondary concern raised in the issue: these data-binding entities are not
  documented anywhere. A user has no way to discover that `llm_response` is the
  variable exposed inside `state_updates`, or that `prompts`/`llms`/`skills`
  are top-level CEL variables.

## Scope

1. Add a top-level `data:` mapping to the declarative config schema. It accepts
   an arbitrary nested mapping of static values.
2. Expose the mapping to every CEL expression in the agent as the top-level
   `data` variable, accessed via `data.key` / `data["key"]` (mirrors how
   `prompts`, `llms`, and `skills` are exposed).
3. Environment-variable interpolation already runs over the whole YAML before
   parsing, so `${VAR}` placeholders inside `data` values keep working with no
   extra code.
4. Document the new `data` section **and** add a "Data Binding Reference"
   table to the declarative-agents docs that enumerates every variable exposed
   to CEL — including the runtime-scoped `llm_response`.
5. Mirror the docs into `skills/sherma/references/` and update
   `skills/sherma/SKILL.md` (top-level keys, CEL cheat sheet, gotchas).

## Non-goals

- No templating/interpolation of one `data` value from another.
- No JSON-Schema validation of the `data` mapping (it is free-form by design).

## Chat Iterations

- (none yet)
