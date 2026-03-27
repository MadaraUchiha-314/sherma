# Task 46: Custom Node Type via Hooks

**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/46

## Description

Add a `custom` node type and a new `node_execute` hook that fires only for custom nodes. This replaces the originally proposed `python_node` (with `import_path` in YAML) to preserve the declarative nature of the agent definition. All custom logic lives in hook executors, not in node args.

## Requirements

1. New `custom` node type in declarative YAML with optional `metadata` dict in args
2. New `node_execute` hook (`NODE_EXECUTE`) that fires only for `custom` nodes
3. `NodeExecuteContext` with `node_context`, `node_name`, `state`, and `result` (starts as `{}`)
4. Lifecycle: `node_enter` → `node_execute` → `node_exit`
5. Returned `result` dict merged into state (same semantics as `data_transform`)
6. Works with all edge types (static, conditional) like any other node
7. `node_enter`, `node_exit`, `on_node_error` fire normally
8. Support in both local (`HookExecutor`) and remote (`HookHandler`/`RemoteHookExecutor`) hooks
9. Validation: warn if a `custom` node exists but no hooks are registered

## Chat Iterations

### Iteration 1: Hooks instead of python_node

The original issue proposed a `python_node` type with `import_path` pointing to an async function. After discussion, we decided this breaks the declarative construct. Instead, we use a `custom` node type + a dedicated `node_execute` hook. Python logic lives in hook executors (already the established extensibility mechanism), keeping YAML purely declarative.

### Iteration 2: Dedicated node_execute hook

Rather than reusing `node_exit` to inject custom logic (which overloads its semantics), we introduce a dedicated `node_execute` hook that only fires for `custom` nodes. This keeps each hook's purpose explicit:
- `node_enter` = before execution
- `node_execute` = the work (custom nodes only)
- `node_exit` = after execution
