# Plan: Expose Registries in NodeExecuteContext

**Task:** [tasks/64-expose-registries-in-nodeexecutecontext.md](../64-expose-registries-in-nodeexecutecontext.md)
**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/64

---

## 1. Background & Analysis

### Current State
- `NodeExecuteContext` (`sherma/hooks/types.py:222-232`) exposes `node_context`, `node_name`, `state`, and `result`.
- The `custom` node builder (`sherma/langgraph/declarative/nodes.py:1160-1225`) has no reference to the `RegistryBundle` — `_build_node` in `agent.py` builds the `NodeContext` with only `config`, `node_def`, `hook_manager`, and `extra`.
- Custom-node hooks therefore cannot reach chat models, tools, prompts, skills, or sub-agents without the agent author wiring them in out-of-band (e.g. capturing closures at startup), which couples hook logic to initialization and breaks the self-contained, declarative model.

### What we need
- Custom-node `node_execute` hooks should be able to invoke chat models, look up tools, resolve prompts, and load skills at execution time using only the hook context.
- The registries are intentionally **local-only** — they contain live Python objects (chat models, ToolEntity instances) and are not serializable over JSON-RPC.

---

## 2. Design

### Option A — Expose registries on `NodeContext`
- `NodeContext` already aggregates per-node dependencies (`config`, `node_def`, `hook_manager`, `extra`). Adding `registries: RegistryBundle | None = None` keeps all node-level dependencies in one place, and every hook that already receives a `node_context` automatically gets the registries via `ctx.node_context.registries`.

### Option B — Expose registries only on `NodeExecuteContext`
- Matches the issue text literally ("add a `registries` field to the context"). Cleanly scoped to custom nodes.

### Chosen approach — both
- Add `registries: RegistryBundle | None = None` to `NodeContext` so the registries flow into all node builders uniformly (future-proof, and avoids a second plumbing path).
- Additionally add `registries: RegistryBundle | None = None` to `NodeExecuteContext` as a **direct, top-level accessor** so custom-node hook authors can write `ctx.registries.llm_registry` instead of `ctx.node_context.registries.llm_registry`. This matches the issue request and is the ergonomic surface advertised in the docs.
- `build_custom_node` populates `NodeExecuteContext.registries` from `ctx.registries` when firing the `node_execute` hook.

### Serialization
- `RegistryBundle` contains live chat-model instances and Python registry objects — it cannot be JSON-serialized.
- Add `"registries"` to `_NON_SERIALIZABLE_FIELDS` in `sherma/hooks/serialization.py` so `RemoteHookExecutor` strips it on the wire and re-attaches the local reference on the way back. Remote hooks will not receive live registries (they are by definition in a different process), and that is documented.

### Typing
- `RegistryBundle` is imported only under `TYPE_CHECKING` in both `sherma/hooks/types.py` and `sherma/langgraph/declarative/nodes.py` to avoid a runtime import cycle (`sherma.registry` depends on nothing in `sherma.hooks`, but keeping hooks import-light is consistent with the existing pattern for `NodeContext`).

---

## 3. Implementation Steps

### Step 1 — `NodeContext` registries field
- File: `sherma/langgraph/declarative/nodes.py`
- Add `registries: RegistryBundle | None = None` to the `NodeContext` dataclass.
- Add `RegistryBundle` to the `TYPE_CHECKING` imports.

### Step 2 — `NodeExecuteContext` registries field
- File: `sherma/hooks/types.py`
- Add `registries: "RegistryBundle | None" = None` to `NodeExecuteContext`.
- Add `RegistryBundle` to the `TYPE_CHECKING` imports.
- Update the docstring to mention registries.

### Step 3 — Pass registries when building `NodeContext`
- File: `sherma/langgraph/declarative/agent.py`
- In `_build_node`, pass `registries=self._registries` when constructing `NodeContext`.

### Step 4 — Populate `NodeExecuteContext.registries` in custom node builder
- File: `sherma/langgraph/declarative/nodes.py`
- In `build_custom_node`, pass `registries=_ctx.registries` when constructing `NodeExecuteContext`.

### Step 5 — Mark `registries` non-serializable
- File: `sherma/hooks/serialization.py`
- Add `"registries"` to `_NON_SERIALIZABLE_FIELDS`.

### Step 6 — Tests (`tests/langgraph/declarative/test_nodes.py`)
- `test_custom_node_registries_exposed` — registry bundle supplied via `NodeContext` appears on `NodeExecuteContext.registries` inside the hook.
- `test_custom_node_registries_default_none` — when no registries are supplied, `ctx.registries is None`.
- `test_custom_node_hook_uses_llm_registry` — hook looks up a chat model via `ctx.registries.chat_models` and invokes it, verifying the returned AIMessage is reflected in the node result.
- `test_custom_node_hook_uses_tool_registry` — hook awaits `ctx.registries.tool_registry.get(...)` with a pre-populated entity.

### Step 7 — Serialization test (`tests/hooks/test_hooks_serialization.py` or existing file)
- Verify that `serialize_context(NodeExecuteContext(...))` omits the `registries` field and that `deserialize_into_context` re-attaches the original registries reference.

### Step 8 — Docs
- `docs/hooks.md` — update the `NodeExecuteContext` dataclass snippet to include `registries: RegistryBundle | None` and describe it.
- `docs/declarative-agents.md` — extend the `custom` node section with a short example showing a custom node that invokes a chat model via `ctx.registries`.
- `docs/api-reference.md` — the bulleted list does not need changes, but add a one-line note under `NodeExecuteContext` if there is an expanded reference.

### Step 9 — Skill references (mirror docs)
- `skills/sherma/references/hooks.md`
- `skills/sherma/references/declarative-agents.md`
- `skills/sherma/references/api-reference.md`

### Step 10 — `skills/sherma/SKILL.md`
- Add a short gotcha / capability note under custom nodes: registries are accessible from `node_execute` hooks via `ctx.registries`.

### Step 11 — Task + plan bookkeeping
- Create `tasks/64-expose-registries-in-nodeexecutecontext.md`.
- Confirm plan file (this file) is saved.
- Document any iterations in the Chat Iterations / Plan Revisions sections.

### Step 12 — Lint, format, type-check, tests
- `uv run ruff format .`
- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -m "not integration"`

---

## Plan Revisions

_(none yet)_
