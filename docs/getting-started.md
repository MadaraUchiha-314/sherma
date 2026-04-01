# Getting Started

## Installation

```bash
pip install sherma
```

For running examples that use OpenAI models:

```bash
pip install sherma[examples]
```

If developing from source:

```bash
git clone https://github.com/MadaraUchiha-314/sherma.git
cd sherma
uv sync
```

## Prerequisites

- Python 3.13+
- An OpenAI API key (or any OpenAI-compatible LLM provider) for LLM-backed agents

## Your First Agent: Programmatic

The programmatic approach gives you full control over graph construction using LangGraph directly, with sherma providing the registry, A2A integration, and lifecycle management.

```python
import asyncio
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from sherma import LLM, Prompt, Tool
from sherma.langgraph.agent import LangGraphAgent
from sherma.registry.base import RegistryEntry
from sherma.registry.llm import LLMRegistry
from sherma.registry.prompt import PromptRegistry
from sherma.registry.tool import ToolRegistry


class WeatherAgent(LangGraphAgent):
    api_key: str
    prompt_registry: PromptRegistry
    llm_registry: LLMRegistry
    tool_registry: ToolRegistry

    async def get_graph(self) -> CompiledStateGraph:
        # Retrieve entities from registries
        llm_entity = await self.llm_registry.get("openai-gpt-4o-mini")
        prompt_entity = await self.prompt_registry.get("weather-system-prompt")

        llm = ChatOpenAI(model=llm_entity.model_name, api_key=self.api_key)
        llm_with_tools = llm.bind_tools([get_weather])

        async def call_model(state: MessagesState) -> dict:
            system = {"role": "system", "content": prompt_entity.instructions}
            response = await llm_with_tools.ainvoke([system, *state["messages"]])
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode([get_weather]))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        return graph.compile()
```

To use the agent, send it A2A messages:

```python
from a2a.types import Message, Part, Role, TextPart

message = Message(
    message_id="user-1",
    parts=[Part(root=TextPart(text="What's the weather in Tokyo?"))],
    role=Role.user,
)

async for event in agent.send_message(message):
    if isinstance(event, Message):
        for part in event.parts:
            if part.root.kind == "text":
                print(part.root.text)
```

## Your First Agent: Declarative

The same agent can be defined entirely in YAML:

```yaml
# weather-agent.yaml
prompts:
  - id: weather-system-prompt
    version: "1.0.0"
    instructions: >
      You are a helpful weather assistant.
      Use the get_weather tool to look up current weather
      for any city the user asks about.

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
            prompt: 'prompts["weather-system-prompt"]["instructions"]'
            tools:
              - id: get_weather
                version: "1.0.0"
            state_updates:
              messages: '[llm_response]'

      edges:
        - source: agent
          target: __end__
```

Load and run in Python:

```python
from sherma import DeclarativeAgent

agent = DeclarativeAgent(
    id="weather-agent",
    version="1.0.0",
    yaml_path="weather-agent.yaml",
)

# Same send_message interface as the programmatic agent
async for event in agent.send_message(message):
    print(event)
```

When a `call_llm` node declares tools, sherma automatically injects a `tool_node` after it with the correct conditional edges. You don't need to wire tool execution manually.

## Serving with A2A

To expose any sherma agent as an A2A server:

```python
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentCapabilities

from sherma.a2a import ShermaAgentExecutor

agent = DeclarativeAgent(
    id="weather-agent",
    version="1.0.0",
    yaml_path="weather-agent.yaml",
)

executor = ShermaAgentExecutor(agent)
handler = DefaultRequestHandler(agent_executor=executor)
card = AgentCard(
    name="Weather Agent",
    description="Looks up weather for any city",
    url="http://localhost:8000",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
)
app = A2AStarletteApplication(agent_card=card, http_handler=handler)
```

## Next Steps

- [Core Concepts](concepts.md) -- understand entities, registries, and versioning
- [Declarative Agents](declarative-agents.md) -- full YAML schema reference
- [Multi-Agent](multi-agent.md) -- sub-agent orchestration and agent-as-tool wrapping
- [Skills](skills.md) -- progressive skill disclosure
- [Hooks](hooks.md) -- lifecycle hooks for observability and control
