# Plan: Custom Node Type via Hooks

**Task:** [tasks/46-custom-node-type-via-hooks.md](../46-custom-node-type-via-hooks.md)
**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/46

---

## 1. Background & Analysis

### Current State
- 6 built-in node types: `call_llm`, `tool_node`, `call_agent`, `data_transform`, `set_state`, `interrupt`
- 17 hook types covering lifecycle events (enter/exit, before/after operations, errors)
- No hook exists for replacing a node's core execution logic
- `node_enter` return values are discarded in all node builders
- `node_exit` can modify results but fires after built-in logic has already run

### Why not `python_node`
The originally proposed `python_node` with `import_path` in YAML breaks the declarative model by embedding imperative Python references in node definitions. Hooks are already the mechanism for injecting Python — we should lean into that.

---

## 2. Design

### New Hook: `node_execute`

- Fires **only** for `custom` nodes, between `node_enter` and `node_exit`
- Context: `NodeExecuteContext(node_context, node_name, state, result)`
- `result` starts as `{}`, hook populates it
- Returned dict is merged into state (same as `data_transform`)

### New Node Type: `custom`

```yaml
nodes:
  - name: summarize_if_needed
    type: custom
    args:
      metadata:  # optional, passed through to hooks via node_context
        description: "Summarize long conversations"
```

### Lifecycle

```
node_enter → node_execute → node_exit
```

### Error Handling

- `on_node_error` fires if `node_execute` raises
- Fallback routing supported (add `custom` to `_FALLBACK_ALLOWED`)

---

## 3. Implementation Steps

### Step 1: Schema changes (`sherma/langgraph/declarative/schema.py`)
- Add `CustomArgs` model with optional `metadata: dict[str, Any]`
- Add `"custom"` to `NodeDef.type` literal union
- Add `CustomArgs` to `NodeDef.args` union

### Step 2: Hook types (`sherma/hooks/types.py`)
- Add `NODE_EXECUTE` to `HookType` enum
- Add `NodeExecuteContext` dataclass

### Step 3: Hook executor protocol (`sherma/hooks/executor.py`)
- Add `node_execute` method to `HookExecutor` protocol
- Add default no-op `node_execute` to `BaseHookExecutor`

### Step 4: Hook handler for remote hooks (`sherma/hooks/handler.py`)
- Add `node_execute` to `HookHandler` base class

### Step 5: Remote hook executor (`sherma/hooks/remote.py`)
- Add `node_execute` support to `RemoteHookExecutor`

### Step 6: Node builder (`sherma/langgraph/declarative/nodes.py`)
- Add `build_custom_node()` function
- Flow: `node_enter` → `node_execute` hook → `node_exit` → return result
- Error handling: catch exceptions, fire `on_node_error`, support fallback

### Step 7: Graph compilation (`sherma/langgraph/declarative/agent.py`)
- Add `"custom"` case to node building dispatch

### Step 8: Loader validation (`sherma/langgraph/declarative/loader.py`)
- Add `"custom"` to `_FALLBACK_ALLOWED`
- Add validation: warn if `custom` node exists but no hooks registered

### Step 9: Tests
- Custom node with hook providing result
- Custom node without hook (returns empty dict)
- Custom node with conditional edges
- Custom node with `on_error` and fallback
- Custom node with remote hook
- Multiple custom nodes with different hooks dispatching by `node_name`

### Step 10: Update docs and skill references
- Update `docs/` with custom node type and `node_execute` hook documentation
- Update `skills/sherma/references/` (copies of docs)
- Update `skills/sherma/SKILL.md` if it covers node types or hooks

---

## Plan Revisions

_(none yet)_
