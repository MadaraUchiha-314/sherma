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

CEL hooks resolve the underlying need. A `node_exit` CEL hook with
`state_updates`, scoped to the interrupt node with a `when` guard, performs
exactly the declarative state assignment #68 wants — no extra `set_state` node:

```yaml
hooks:
  - cel:
      node_exit:
        when: 'node_type == "interrupt" && node_name == "get_approval"'
        state_updates:
          response_type: >
            result["messages"][size(result["messages"]) - 1]["additional_kwargs"]["action"]
```

Why `node_exit` (and `result`) rather than `after_interrupt`/`state`: the
interrupt node builds its returned state delta from the resume value and
returns *that* dict — not its in-flight `ctx.state`. The resumed message is
therefore in `ctx.result["messages"]` at `node_exit`, and `state_updates`
merge into `result` (the node's actual return value), so they persist into the
graph state. State mutations on `after_interrupt` would not persist because
the node ignores `ctx.state` for its return. `state_updates` in a CEL hook are
defined to target `result` whenever the context exposes one for exactly this
reason.

Difference in shape: #68 wanted the update inline on the node's `args`, whereas
a CEL hook is declared at the agent level and scoped with `when`. Functionally
it covers the use case (and generalises to every other lifecycle point), so
#68 can be closed as resolved-by-#82. Inlining `state_updates` on interrupt
`args` remains possible as a separate ergonomic nicety but is no longer
required to unblock the use case.

## Chat Iterations

_None yet._
