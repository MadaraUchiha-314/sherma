# Multi-Agent Orchestration

sherma supports multi-agent systems where a supervisor agent delegates work to sub-agents. Sub-agents are automatically wrapped as LangGraph tools, so the supervisor's LLM decides when and how to invoke them through standard tool calling.

## How It Works

1. You declare sub-agents in the YAML config under `sub_agents`
2. Each sub-agent is registered in the agent registry and wrapped as a LangGraph tool
3. A `call_llm` node with `use_sub_agents_as_tools` binds these tools to the LLM (use `true`/`all` for all sub-agents, or a list to select specific ones)
4. The LLM invokes sub-agents through tool calls; sherma handles the A2A message plumbing
5. Tool nodes and conditional edges are auto-injected, just like regular tools

## Declaring Sub-Agents

Sub-agents can be sourced in four ways:

### From a YAML file

The simplest approach -- point directly to another declarative agent's YAML. The parent's `http_async_client` (and API keys) are automatically forwarded.

```yaml
sub_agents:
  - id: weather-agent
    version: "1.0.0"
    yaml_path: weather-agent.yaml  # Relative to the parent YAML file
```

Relative `yaml_path` values are resolved against the parent YAML file's directory (i.e., `base_path`). The sub-agent automatically derives its own `base_path` from its resolved YAML path, so nested sub-agents chain correctly. See [Path Resolution](declarative-agents.md#path-resolution).

### From a Python module

Import a pre-constructed `Agent` instance from a Python module:

```yaml
sub_agents:
  - id: weather-agent
    version: "1.0.0"
    import_path: my_agents.weather_agent
```

The import path should resolve to an `Agent` instance (not a class).

### From a remote URL

Connect to an A2A-compatible agent running elsewhere:

```yaml
sub_agents:
  - id: remote-agent
    version: "1.0.0"
    url: https://remote-agent.example.com
```

### Pre-registered in the registry

If neither `yaml_path`, `import_path`, nor `url` is provided, the agent is expected to already exist in the agent registry:

```yaml
sub_agents:
  - id: my-agent
    version: "1.0.0"
```

This is useful when you register agents programmatically before building the supervisor.

## Supervisor YAML

A supervisor agent uses `use_sub_agents_as_tools` on its `call_llm` node. Use `true` (or `all`) to bind all declared sub-agents, or provide a list of specific `id`/`version` refs:

```yaml
prompts:
  - id: supervisor-prompt
    version: "1.0.0"
    instructions: >
      You are a travel planner. Use the weather agent to check conditions
      at the destination before making recommendations.

llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai
    model_name: gpt-4o-mini

sub_agents:
  - id: weather-agent
    version: "1.0.0"
    yaml_path: weather-agent.yaml  # Resolved relative to this YAML file

agents:
  travel-planner:
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: planner
      nodes:
        - name: planner
          type: call_llm
          args:
            llm:
              id: openai-gpt-4o-mini
              version: "1.0.0"
            prompt: 'prompts["supervisor-prompt"]["instructions"]'
            use_sub_agents_as_tools: true
            state_updates:
              messages: '[llm_response]'

      edges:
        - source: planner
          target: __end__
```

The `tool_node` and conditional edges are auto-injected, so you only need to define the `call_llm` node.

### Selecting specific sub-agents

Instead of binding all sub-agents, you can select a subset by providing a list of `id`/`version` refs:

```yaml
use_sub_agents_as_tools:
  - id: weather-agent
    version: "1.0.0"
```

This is useful when you have many sub-agents declared but a particular node should only access some of them.

## Agent Input Schemas

If a sub-agent defines an `input_schema` (a Pydantic model), the generated tool accepts two arguments:

- `request: str` -- natural language request text
- `agent_input: <Schema>` -- structured input matching the agent's schema

If no `input_schema` is defined, the tool accepts only `request: str`.

## Using Building Blocks Without Declarative Agents

The multi-agent utilities described here are not exclusive to `DeclarativeAgent`. If you are building agents directly with `LangGraphAgent` or plain LangGraph, you can use these building blocks independently:

### `agent_to_langgraph_tool`

Wrap any `Agent` instance as a LangGraph `BaseTool`. This is the core primitive that the declarative `sub_agents` config builds on, and you can use it directly in your own graph construction code:

```python
from sherma.langgraph.tools import agent_to_langgraph_tool

# Wrap any sherma Agent (local, remote, declarative) as a tool
weather_tool = agent_to_langgraph_tool(weather_agent)

# Use it in your own LangGraph graph
llm_with_tools = chat_model.bind_tools([weather_tool, other_tool])
```

### Registries and tool conversion

The registry system, tool wrapping helpers (`from_langgraph_tool`, `to_langgraph_tool`), and A2A message conversion are all standalone utilities. You can register agents in an `AgentRegistry`, resolve them by ID/version, and wrap them as tools -- all without writing any YAML:

```python
from sherma.registry.base import RegistryEntry
from sherma.registry.bundle import RegistryBundle
from sherma.langgraph.tools import agent_to_langgraph_tool, from_langgraph_tool

bundle = RegistryBundle()

# Register an agent
await bundle.agent_registry.add(
    RegistryEntry(id="weather-agent", version="1.0.0", instance=my_agent)
)

# Resolve and wrap as a tool
agent = await bundle.agent_registry.get("weather-agent")
tool = agent_to_langgraph_tool(agent)

# Register the tool for use elsewhere
sherma_tool = from_langgraph_tool(tool)
await bundle.tool_registry.add(
    RegistryEntry(id=sherma_tool.id, version="1.0.0", instance=sherma_tool)
)
```

This applies broadly across sherma -- hooks, skills, schema helpers, and message converters are all usable as composable building blocks in your own `LangGraphAgent` subclass or raw LangGraph code.

## Running the Example

The `examples/multi_agent/` directory contains a complete working example:

```
examples/multi_agent/
├── weather_agent.yaml       # Sub-agent: looks up weather
├── supervisor_agent.yaml    # Supervisor: travel planner
└── main.py                  # Entry point
```

```bash
uv run python examples/multi_agent/main.py "Plan a trip to Tokyo"
```

## Tenant Propagation

When a `DeclarativeAgent` has a `tenant_id`, it propagates to all entities it creates -- including sub-agents declared via `yaml_path`. This means sub-agents inherit the parent's tenant scope automatically.

```python
agent = DeclarativeAgent(
    id="travel-planner",
    version="1.0.0",
    yaml_path="supervisor.yaml",
    tenant_id="acme-corp",  # Propagated to sub-agents and all registry entries
)
```
