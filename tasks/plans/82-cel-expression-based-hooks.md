# Plan 82: CEL expression based hooks in declarative YAML

Plan for `tasks/82-cel-expression-based-hooks.md`.

## Design

### YAML surface

`HookDef` gains a fourth mutually-exclusive source, `cel`, whose value is a map
of `hook_name -> CELHookSpec`:

```yaml
hooks:
  - cel:
      before_llm_call:
        set:
          system_prompt: 'system_prompt + "\n\nBe concise."'
      after_interrupt:
        when: 'node_name == "get_approval"'
        state_updates:
          response_type: >
            state.messages[size(state.messages) - 1]["additional_kwargs"]["action"]
```

`CELHookSpec`:

* `when: str | None` — optional CEL boolean guard.
* `state_updates: dict[str, str]` — CEL expressions merged into `ctx.state`.
* `set: dict[str, str]` — CEL expressions assigned to other writable fields.

At least one of `state_updates` / `set` must be present.

### Activation model

For each hook invocation the executor builds a `CelEngine` activation:

* `state` = `ctx.state` (or `{}` for state-less contexts) → reachable as
  `state.<field>`.
* every other dataclass field is exposed at the top level by its own name
  (`node_name`, `system_prompt`, `response`, `value`, `config`, ...), converted
  through the existing `_python_to_cel` path (so messages/response become
  map-like and metadata is readable, exactly as in `call_llm` CEL).

`when` is evaluated first; a false result short-circuits to a no-op (return
`None`). Then `state_updates` are evaluated and merged into a copy of
`ctx.state`; then `set` fields are evaluated and assigned onto `ctx`. The
(mutated) context is returned so it threads through the `HookManager` chain.

### Writable-field rules

Writable fields per hook = dataclass fields − non-serializable
(`node_context`, `agent`, `chat_model`, `registries`) − complex
(`messages`, `response`, `tools`, `tool_calls`) − `error` − `state` (handled by
`state_updates`). `on_chat_model_create` is disallowed entirely (returns a
Python object). State-bearing = contexts that have a `state` field.

These rules are computed by introspecting the dataclasses + reusing the
constants already in `sherma/hooks/serialization.py`, so they stay in sync.

## Steps

1. **`sherma/hooks/cel_executor.py`** (new)
   - `writable_fields(hook_name)` / `hook_has_state(hook_name)` helpers driven
     by dataclass introspection + serialization constants. Export the set of
     CEL-capable hook names.
   - `CELHookExecutor(BaseHookExecutor)`: holds the parsed spec + a `CelEngine`.
     Override each CEL-capable hook method with a one-line delegate to a shared
     `_apply(hook_name, ctx)` that builds the activation, runs `when`,
     `state_updates`, and `set`, and returns the context.

2. **`sherma/langgraph/declarative/schema.py`**
   - Add `CELHookSpec` model with `when` / `state_updates` / `set` and a
     validator requiring at least one of `state_updates` / `set`.
   - Add `cel: dict[str, CELHookSpec] | None` to `HookDef`; extend the
     mutual-exclusion validator to include it. Validate hook names, the
     `on_chat_model_create` exclusion, `state_updates` only on state-bearing
     hooks, and `set` keys ⊆ writable fields.

3. **`sherma/langgraph/declarative/loader.py`**
   - `populate_hooks` gains a `cel_extra_vars` parameter. For a `cel` def,
     build a `CelEngine(extra_vars=cel_extra_vars)` and register a
     `CELHookExecutor`.

4. **`sherma/langgraph/declarative/agent.py`**
   - Pass `self._build_cel_extra_vars(config)` into `populate_hooks`.

5. **Exports** — surface `CELHookExecutor` from `sherma/hooks/__init__.py` and
   `sherma/__init__.py`.

6. **Example** — `examples/declarative_cel_hooks_agent/` (YAML + `main.py`)
   re-implementing the approval-tagging flow with a CEL `after_interrupt` /
   `node_exit` hook instead of the Python `ApprovalTaggingHook`, demonstrating
   the #68 use case.

7. **Tests**
   - `tests/hooks/test_cel_executor.py` — activation, `when` guard,
     `state_updates`, `set`, no-op when hook absent.
   - schema tests in `tests/langgraph/declarative/test_schema.py` — valid spec,
     mutual exclusion, bad hook name, `on_chat_model_create` rejected, bad
     `set` field, `state_updates` on state-less hook rejected.
   - loader test — `cel` def registers a `CELHookExecutor`.
   - `tests/integration/test_declarative_cel_hooks_agent.py` — end-to-end
     interrupt/resume with state updated by the CEL hook.

8. **Docs + skill** (CLAUDE.md mandate)
   - `docs/hooks.md`: new "CEL Hooks" section (surface, activation, writable
     fields, #68 note). `docs/declarative-agents.md`: hook-source list +
     example. `docs/api-reference.md`: `CELHookExecutor`. Mirror all three into
     `skills/sherma/references/`. Update `skills/sherma/SKILL.md` (hook source
     list / quick reference / gotchas).

9. **Quality gates** — `ruff check`, `ruff format`, `pyright`, `pytest -m "not
   integration"`; commit and push to `claude/youthful-gates-HB5KT`.

## Plan Revisions

### state_updates target `result`, not `state` (during implementation)

LangGraph nodes persist state by *returning* a delta dict, not by mutating the
`state` dict passed in. So a CEL hook that mutates `ctx.state` does not persist.
`state_updates` were redefined to merge into `ctx.result` whenever the context
exposes one (`node_exit`, `node_execute`, `after_tool_call`,
`after_agent_call`, `after_graph_invoke`) — that dict is the node's return
value — and to fall back to `ctx.state` (in-flight, visible to later hooks)
otherwise. Validation (`hook_supports_state_updates`) allows `state_updates`
when the context has `result` **or** `state`. The #68 example consequently
uses a `node_exit` hook on the interrupt node (reading `result["messages"]`),
which is also what the existing Python `ApprovalTaggingHook` does.

A new `extra` parameter was added to `CelEngine.evaluate_bool` so the `when`
guard can reference top-level context fields (e.g. `node_name`, `node_type`).
