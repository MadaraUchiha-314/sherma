# Task 82: CEL expression based hooks in declarative YAML

**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/82

## Description

Today a hook can be wired into a declarative agent in three ways:

* `import_path` — a local Python `HookExecutor`;
* `url` — a remote JSON-RPC 2.0 hook server;
* `mcp` — an MCP-based hook server.

All three require code (or an external service) to be written. This task adds a
fourth, fully-declarative option: define hook behaviour directly in the YAML
using **CEL expressions**. A CEL hook evaluates expressions against the hook
context and produces the same effect as the return value of a Python hook —
most importantly, it can update graph `state` and other modifiable context
fields without leaving the YAML.

## Requirements

1. Add `cel` as a fourth, mutually-exclusive source on `HookDef`.
2. A CEL hook declares, per hook type, a spec with:
   * `when` (optional) — a CEL boolean guard; the hook is a no-op when false.
   * `state_updates` (optional) — a map of `state_key -> CEL expression`;
     results are merged into `ctx.state`.
   * `set` (optional) — a map of `field -> CEL expression` for other modifiable
     scalar/dict context fields (e.g. `system_prompt`, `config`, `result`).
3. Every context field is readable in the CEL activation at the top level
   (`node_name`, `system_prompt`, `response`, `value`, ...); graph `state`
   fields are reachable via `state.<field>` as elsewhere in sherma.
4. Reuse the existing `CelEngine` (so `default(...)`, custom functions,
   message-metadata access, `data`/`prompts`/`llms` vars all work).
5. Validate at config-load time: known hook names, no `on_chat_model_create`
   (it returns a Python object), `state_updates` only on state-bearing
   contexts, and `set` keys restricted to writable fields per hook.
6. Update docs and the skill references.
7. Add unit tests, schema tests, and an integration test plus a worked example.

## Evaluation: does this fix issue #68?

Issue #68 asks for `state_updates` on **interrupt nodes** so that, after an
interrupt resumes, a state field can be set declaratively (its example sets
`response_type` from the resumed message's `additional_kwargs`).

CEL hooks resolve the underlying need. A `node_exit` CEL hook whose expression
returns `{"result": {...}}`, scoped to the interrupt node with a ternary guard,
performs exactly the declarative state assignment #68 wants — no extra
`set_state` node:

```yaml
hooks:
  - type: cel
    expressions:
      node_exit: >
        node_type == "interrupt" && node_name == "get_approval"
          ? {"result": {"response_type":
              result["messages"][size(result["messages"]) - 1]["additional_kwargs"]["action"]}}
          : {}
```

Why `node_exit` (and `result`) rather than `after_interrupt`/`state`: the
interrupt node builds its returned state delta from the resume value and
returns *that* dict — not its in-flight `ctx.state`. The resumed message is
therefore in `ctx.result["messages"]` at `node_exit`, and a returned
`{"result": {...}}` merges into `result` (the node's actual return value), so
it persists into the graph state. State mutations on `after_interrupt` would
not persist because the node ignores `ctx.state` for its return. Dict-valued
fields like `result` merge for exactly this reason — returning only
`response_type` preserves the resumed message already in `result["messages"]`.

Difference in shape: #68 wanted the update inline on the node's `args`, whereas
a CEL hook is declared at the agent level and scoped with a ternary guard.
Functionally it covers the use case (and generalises to every other lifecycle
point), so #68 can be closed as resolved-by-#82. Inlining `state_updates` on
interrupt `args` remains possible as a separate ergonomic nicety but is no
longer required to unblock the use case.

## Chat Iterations

### 2026-06-04: PR #83 review — redesign to single expression + `type:` discriminator

Owner review on PR #83 raised two design questions, resolved as:

1. **Drop the `when` / `state_updates` / `set` keywords.** The issue's intent
   was that a CEL expression evaluates to the hook's *return value*. So each
   hook now maps to a **single CEL expression** that returns a map of
   `context_field -> new value` (the return delta); `{}` is a no-op and a guard
   is just a CEL ternary returning `{}`. Dict-valued fields (`state`, `result`,
   …) merge; scalar fields are replaced. Merge semantics preserve the original
   (non-CEL-convertible) message objects in `result["messages"]`.
2. **Use a `type:` discriminator for every hook entry.** The `hooks:` array is
   now a discriminated union — each entry is `{type: import_path|url|mcp|cel,
   …}`. `HookDef` became `Annotated[ImportPathHookDef | URLHookDef | MCPHookDef
   | CELHookDef, Field(discriminator="type")]`. This is a **breaking change** to
   existing `import_path`/`url`/`mcp` configs (accepted by the owner); all
   in-repo examples, tests, and docs were migrated.

`CELHookSpec` was replaced by `CELHookDef.expressions: dict[str, str]`;
`hook_supports_state_updates` was removed; `writable_fields` now includes
`state`/`result`.

### 2026-06-04: PR #83 review — drop the `expressions` indirection

Owner follow-up: the `expressions:` sub-key is unnecessary. Flattened it — hook
names now sit directly on the `cel` entry alongside `type` (no hook is named
`type`, so there's no collision). `CELHookDef` uses `model_config =
ConfigDict(extra="allow")`; the extra keys are validated as `hook_name -> CEL
string` and surfaced via an `expressions` property (so the loader/executor are
unchanged). Example/tests/docs updated.

```yaml
hooks:
  - type: cel
    before_llm_call: '{"system_prompt": system_prompt + "..."}'
```
