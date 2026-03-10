# Plan: Hooks System for Sherma

## Context

Developers using sherma's LangGraph and Declarative Agents need programmatic control over agent lifecycle events (LLM calls, tool calls, agent calls, skill loading, node transitions, interrupts). This hooks system gives them that control via a `HookExecutor` protocol with before/after methods for each lifecycle point, a `HookManager` to orchestrate multiple executors in registration order, and integration into the existing node builder architecture.

## Design

### Core Abstractions

**`HookType` enum** - Identifies each lifecycle hook point (12 total: before/after for llm_call, tool_call, agent_call, skill_load, interrupt, plus node_enter/node_exit).

**Context dataclasses** - Each hook type gets a typed context carrying relevant data. **All context dataclasses include a `node_context: NodeContext` field**, giving every hook access to the full declarative config, node definition, and extras:
- `BeforeLLMCallContext(node_context, node_name, messages, system_prompt, tools, state)` - can modify prompt, tools, messages
- `AfterLLMCallContext(node_context, node_name, response, state)` - can modify/replace LLM response
- `BeforeToolCallContext(node_context, node_name, tool_calls, tools, state)` - can filter/modify tool calls
- `AfterToolCallContext(node_context, node_name, result, state)` - can modify tool results
- `BeforeAgentCallContext(node_context, node_name, input_value, agent, state)` - can modify agent input
- `AfterAgentCallContext(node_context, node_name, result, state)` - can modify agent response
- `BeforeSkillLoadContext(node_context, skill_id, version)` - observe/modify skill load params (node_context may be None when called outside a node, e.g. from skill_tools.py directly)
- `AfterSkillLoadContext(node_context, skill_id, version, content, tools_loaded)` - observe/modify loaded content
- `NodeEnterContext(node_context, node_name, node_type, state)` - observe node entry
- `NodeExitContext(node_context, node_name, node_type, result, state)` - can modify node result
- `BeforeInterruptContext(node_context, node_name, value, state)` - can modify interrupt value
- `AfterInterruptContext(node_context, node_name, value, response, state)` - can modify interrupt response

This gives hook implementations access to `node_context.config` (full DeclarativeConfig), `node_context.node_def` (current node's definition including type, args), and `node_context.extra` (arbitrary extra data).

**`HookExecutor` Protocol** - All 12 async methods, each returning `ContextType | None`. None = no-op.

**`BaseHookExecutor` class** - Default implementation with all methods returning `None`. Users subclass and override only what they need.

**`HookManager` class** - Stores list of executors, provides `register(executor)` and `run_hook(hook_name, ctx) -> ctx`. Chains results: if a hook returns `None`, context passes through; if it returns a value, that replaces the context for the next executor.

### Integration Points

1. **`NodeContext`** (`nodes.py:56`) - Add `hook_manager: HookManager | None = None` field
2. **`LangGraphAgent`** (`agent.py:31`) - Add `hook_manager: HookManager` field with default factory, add `register_hooks()` method
3. **`DeclarativeAgent._build_node()`** (`declarative/agent.py:204`) - Pass `self.hook_manager` into `NodeContext`
4. **Node builders** (`nodes.py`) - Each `build_*_node` function calls hooks via `_ctx.hook_manager` (guarded by `if hooks:`)
5. **Schema** (`schema.py`) - Add `HookDef(import_path: str)` model and `hooks: list[HookDef]` to `DeclarativeConfig`
6. **Loader** (`loader.py`) - Import and register hook executors from YAML `hooks` config
7. **Skill tools** (`skill_tools.py`) - Accept optional `hook_manager`, fire `before/after_skill_load` in `load_skill_md`

### Hook Execution Flow per Node Type

**call_llm**: `node_enter` → `before_llm_call` → LLM invoke → `after_llm_call` → `node_exit`
**tool_node**: `node_enter` → `before_tool_call` → tool execution → `after_tool_call` → `node_exit`
**call_agent**: `node_enter` → `before_agent_call` → agent invoke → `after_agent_call` → `node_exit`
**interrupt**: `node_enter` → `before_interrupt` → `interrupt()` → `after_interrupt` → `node_exit`
**data_transform / set_state**: `node_enter` → execution → `node_exit`
**load_skill_md** (in skill_tools.py): `before_skill_load` → load → `after_skill_load`

## New Files

| File | Purpose |
|------|---------|
| `sherma/hooks/__init__.py` | Re-exports all public hook types |
| `sherma/hooks/types.py` | `HookType` enum + all 12 context dataclasses |
| `sherma/hooks/executor.py` | `HookExecutor` Protocol + `BaseHookExecutor` default impl |
| `sherma/hooks/manager.py` | `HookManager` class |
| `tests/hooks/__init__.py` | Test package |
| `tests/hooks/test_types.py` | Context dataclass construction tests |
| `tests/hooks/test_executor.py` | BaseHookExecutor default behavior tests |
| `tests/hooks/test_manager.py` | Hook chaining, None pass-through, ordering tests |

## Files to Modify

| File | Change |
|------|--------|
| `sherma/langgraph/declarative/nodes.py` | Add `hook_manager` to `NodeContext`, add hook calls in all `build_*_node` functions |
| `sherma/langgraph/agent.py` | Add `hook_manager` field to `LangGraphAgent`, add `register_hooks()` |
| `sherma/langgraph/declarative/agent.py` | Pass `hook_manager` into `NodeContext` in `_build_node()`, handle YAML hook imports in `get_graph()` |
| `sherma/langgraph/declarative/schema.py` | Add `HookDef` model, add `hooks` field to `DeclarativeConfig` |
| `sherma/langgraph/declarative/loader.py` | Add `populate_hooks()` function to import hook classes from YAML config |
| `sherma/langgraph/skill_tools.py` | Accept optional `hook_manager`, fire `before/after_skill_load` in `load_skill_md` |
| `sherma/__init__.py` | Re-export `HookManager`, `HookExecutor`, `BaseHookExecutor`, hook context types |
| `tests/langgraph/declarative/test_nodes.py` | Add tests for hooks firing in node builders |

## Implementation Order

1. Create `sherma/hooks/types.py` - enum + context dataclasses
2. Create `sherma/hooks/executor.py` - Protocol + BaseHookExecutor
3. Create `sherma/hooks/manager.py` - HookManager
4. Create `sherma/hooks/__init__.py` - re-exports
5. Modify `nodes.py` - add `hook_manager` to `NodeContext`, add hook calls in each builder
6. Modify `agent.py` - add `hook_manager` field to `LangGraphAgent`
7. Modify `declarative/agent.py` - wire `hook_manager` into `NodeContext`, handle YAML hooks
8. Modify `schema.py` - add `HookDef` and `hooks` to `DeclarativeConfig`
9. Modify `loader.py` - add hook import logic
10. Modify `skill_tools.py` - add skill load hooks
11. Update `sherma/__init__.py` - re-exports
12. Write all tests
13. Run `uv run ruff check .`, `uv run pyright`, `uv run pytest`

## Verification

1. `uv run ruff check .` - no lint errors
2. `uv run ruff format --check .` - formatting passes
3. `uv run pyright` - no type errors
4. `uv run pytest` - all tests pass (existing + new)
5. `uv run pytest tests/hooks/` - hook infrastructure tests pass
6. `uv run pytest tests/langgraph/declarative/test_nodes.py` - node hook integration tests pass
