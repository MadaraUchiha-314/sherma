# Plan: Wrap Agent as Tool

## Context

For multi-agent orchestration, a supervisor agent needs to invoke sub-agents via LLM tool_calls. Currently, agents can only be invoked through explicit `call_agent` graph nodes, which require static graph wiring. By wrapping agents as tools, the LLM can dynamically choose which sub-agent to call based on the conversation context.

## Implementation

### Step 1: Core wrapper function — `sherma/langgraph/tools.py`

Add `agent_to_langgraph_tool(agent: Agent) -> BaseTool`:

- If agent has **no** `input_schema`: create a `StructuredTool` with a single `request: str` param
- If agent **has** `input_schema`: dynamically build a Pydantic model combining `request: str` + `agent_input: <InputSchema>` using `pydantic.create_model`, then create `StructuredTool` with `args_schema` set to this model

The inner async function:
1. Build A2A `Message` with `TextPart` from `request`
2. If `agent_input` provided, add `DataPart` with `agent_input: True` metadata (reuse `make_schema_data_part` from `sherma/schema.py`)
3. Call `agent.send_message(message)`, collect results
4. Extract text from last `Message` result (same pattern as `build_call_agent_node` in nodes.py:460-465)
5. Return text string

Tool name = `agent.id`. Description from `agent.agent_card.description` or fallback `f"Invoke agent: {agent.id}"`.

### Step 2: Declarative schema — `sherma/langgraph/declarative/schema.py`

Add new model:
```python
class SubAgentDef(BaseModel):
    id: str
    version: str = "*"
    url: str | None = None           # remote agents
    import_path: str | None = None   # local agent classes
```

Add to `DeclarativeConfig`:
```python
sub_agents: list[SubAgentDef] = Field(default_factory=list)
```

Add to `CallLLMArgs`:
```python
use_sub_agents_as_tools: bool = False
```

### Step 3: Register sub-agents in loader — `sherma/langgraph/declarative/loader.py`

In `populate_registries()`, add a section for `config.sub_agents`:

For each `SubAgentDef`:
- If `url` provided: register as remote in `agent_registry` via `RegistryEntry(remote=True, url=...)`
- If `import_path` provided: import the agent class/instance, register in `agent_registry`
- If neither: expect agent already registered externally (skip registration)

After all agents registered, wrap each as a tool:
1. `agent = await registries.agent_registry.get(sub_agent_def.id, ...)`
2. `lg_tool = agent_to_langgraph_tool(agent)`
3. `sherma_tool = from_langgraph_tool(lg_tool)`
4. Register in `tool_registry`

Track sub-agent tool IDs in a list and store on the config or return them.

### Step 4: Wire sub-agent tools into LLM nodes — `sherma/langgraph/declarative/nodes.py`

Add `_resolve_sub_agent_tools()` helper that resolves tools from `tool_registry` by the sub-agent tool IDs stored in `NodeContext.extra["sub_agent_tool_ids"]`.

In `build_call_llm_node`, add handling for `use_sub_agents_as_tools`:
```python
elif args.use_sub_agents_as_tools and tool_registry is not None:
    sub_agent_tool_ids = _ctx.extra.get("sub_agent_tool_ids", [])
    current_tools = await resolve_tools_for_node_async(
        [RegistryRef(id=tid) for tid in sub_agent_tool_ids], tool_registry
    )
```

This is additive — can be combined with other tool sources if needed later.

### Step 5: Pass sub-agent tool IDs through NodeContext — `sherma/langgraph/declarative/agent.py`

In `get_graph()`, after `populate_registries`, collect sub-agent tool IDs:
```python
self._sub_agent_tool_ids = [sa.id for sa in config.sub_agents]
```

In `_build_node()`, pass through `NodeContext.extra`:
```python
ctx = NodeContext(
    config=config,
    node_def=node_def,
    hook_manager=...,
    extra={"sub_agent_tool_ids": self._sub_agent_tool_ids},
)
```

### Step 6: Fix `call_agent` bug — `sherma/langgraph/declarative/agent.py:264-269`

Change from `tool_registry.get(...)` to `agent_registry.get(...)`:
```python
agent = await self._registries.agent_registry.get(
    ca_args.agent.id, ca_args.agent.version
)
```

### Step 7: Validation — `sherma/langgraph/declarative/loader.py`

In `validate_config()`:
- If `use_sub_agents_as_tools: true` on any `call_llm` node, verify `config.sub_agents` is not empty
- Ensure a `tool_node` exists when `use_sub_agents_as_tools` is used (same as other tool flags)

### Step 8: Tests

New file `tests/langgraph/test_agent_as_tool.py`:
- Test `agent_to_langgraph_tool` with agent without `input_schema` — tool has `request: str` only
- Test `agent_to_langgraph_tool` with agent with `input_schema` — tool has `request` + `agent_input` fields
- Test tool invocation calls `agent.send_message` with correct A2A Message structure (mock agent)
- Test DataPart included when `agent_input` is provided

Extend existing declarative tests:
- Test `SubAgentDef` and updated `DeclarativeConfig` parse correctly
- Test `populate_registries` with `sub_agents` registers agents and tool wrappers

## Files to Modify

| File | Change |
|------|--------|
| `sherma/langgraph/tools.py` | Add `agent_to_langgraph_tool()` |
| `sherma/langgraph/declarative/schema.py` | Add `SubAgentDef`, update `DeclarativeConfig`, update `CallLLMArgs` |
| `sherma/langgraph/declarative/loader.py` | Register sub-agents + wrap as tools in `populate_registries()`, add validation |
| `sherma/langgraph/declarative/agent.py` | Fix `call_agent` bug, pass `sub_agent_tool_ids` via `NodeContext.extra` |
| `sherma/langgraph/declarative/nodes.py` | Handle `use_sub_agents_as_tools` in `build_call_llm_node` |
| `tests/langgraph/test_agent_as_tool.py` | New: core wrapper tests |

## Verification

1. `uv run ruff check .` — lint
2. `uv run ruff format --check .` — format
3. `uv run pyright` — type check
4. `uv run pytest` — all tests pass
5. `uv run pytest tests/langgraph/test_agent_as_tool.py` — new tests pass
