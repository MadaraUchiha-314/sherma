<p align="center">
  <img src="https://cdn.wikimg.net/en/hkwiki/images/7/71/Sherma.png" alt="sherma logo" width="200" />
</p>

# sherma

A Python framework for building LLM-powered agents that bridges [A2A](https://a2a-protocol.org/), [LangGraph](https://www.langchain.com/langgraph), and [Agent Skills](https://agentskills.io/) through a unified, declarative interface.

## What is sherma?

sherma lets you define agents in YAML with [Common Expression Language (CEL)](https://cel.dev/) for dynamic logic, while still giving you full programmatic control when you need it. It handles the wiring between protocols so you can focus on agent behavior.

Every utility in sherma -- registries, tool wrapping, hooks, message converters, multi-agent primitives -- works as a standalone building block. You can use them with `DeclarativeAgent` for zero-code YAML agents, with `LangGraphAgent` for custom graph construction, or mix both approaches.

**Core value proposition:**

- **Declarative-first**: Define agents entirely in YAML -- graphs, prompts, tools, skills, and routing logic
- **Multi-agent orchestration**: Compose agents by declaring sub-agents that are automatically wrapped as tools -- the supervisor LLM decides when to delegate
- **Protocol bridge**: Seamlessly connect A2A-compatible agents, LangGraph workflows, MCP tools, and Agent Skills
- **Progressive disclosure**: Skills are discovered and loaded on demand by the LLM, following the [agentskills.io](https://agentskills.io/) specification
- **Lifecycle hooks**: Intercept and modify behavior at every stage -- LLM calls, tool execution, agent invocation, and more

## Documentation

| Document | Description |
| --- | --- |
| [Getting Started](getting-started.md) | Installation, setup, and your first agent |
| [Core Concepts](concepts.md) | Entities, registries, versioning, and the type system |
| [Declarative Agents](declarative-agents.md) | YAML schema reference, node types, edges, and CEL expressions |
| [Multi-Agent](multi-agent.md) | Sub-agent orchestration, agent-as-tool wrapping |
| [Skills](skills.md) | Skill cards, progressive disclosure, MCP and local tool integration |
| [Hooks](hooks.md) | Lifecycle hooks for observability, guardrails, and control flow |
| [A2A Integration](a2a-integration.md) | A2A protocol support, agent executor, message conversion |
| [API Reference](api-reference.md) | Exported classes, functions, and type definitions |

## Quick Example

```yaml
# weather-agent.yaml
prompts:
  - id: weather-prompt
    version: "1.0.0"
    instructions: >
      You are a helpful weather assistant.
      Use the get_weather tool to look up current weather.

llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai
    model_name: gpt-4o-mini

tools:
  - id: get_weather
    version: "1.0.0"
    import_path: my_tools.get_weather

agents:
  weather-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: agent
      nodes:
        - name: agent
          type: call_llm
          args:
            llm:
              id: openai-gpt-4o-mini
              version: "1.0.0"
            prompt:
              - role: system
                content: 'prompts["weather-prompt"]["instructions"]'
              - role: messages
                content: 'messages'
            tools:
              - id: get_weather
                version: "1.0.0"

      edges:
        - source: agent
          target: __end__
```

```python
import asyncio
from sherma import DeclarativeAgent

agent = DeclarativeAgent(
    id="weather-agent",
    version="1.0.0",
    yaml_path="weather-agent.yaml",
)

# Use with A2A messages
from a2a.types import Message, Part, Role, TextPart

msg = Message(
    message_id="1",
    role=Role.user,
    parts=[Part(root=TextPart(text="What's the weather in Tokyo?"))],
)

async def main():
    async for event in agent.send_message(msg):
        print(event)

asyncio.run(main())
```

## Installation

```bash
pip install sherma
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add sherma
```

Requires Python 3.13+.
