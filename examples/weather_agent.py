"""Example weather agent using sherma + LangGraph + ChatOpenAI.

Usage:
    uv run python examples/weather_agent.py "What is the weather in Tokyo?"

Requires:
    - uv sync --extra examples
    - A secrets.json file at the project root with {"openai_api_key": "sk-..."}
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import ConfigDict, Field

from examples.tools import get_weather
from sherma.entities.llm import LLM
from sherma.entities.prompt import Prompt
from sherma.entities.tool import Tool
from sherma.langgraph.agent import LangGraphAgent
from sherma.registry.base import RegistryEntry
from sherma.registry.llm import LLMRegistry
from sherma.registry.prompt import PromptRegistry
from sherma.registry.tool import ToolRegistry

SYSTEM_PROMPT = (
    "You are a helpful weather assistant. "
    "Use the get_weather tool to look up current weather "
    "for any city the user asks about. "
    "Provide a concise, friendly response."
)


class WeatherAgent(LangGraphAgent):
    """A ReAct weather agent backed by ChatOpenAI and Open-Meteo."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str
    prompt_registry: PromptRegistry = Field(default_factory=PromptRegistry)
    llm_registry: LLMRegistry = Field(default_factory=LLMRegistry)
    tool_registry: ToolRegistry = Field(default_factory=ToolRegistry)

    async def _setup_registries(self) -> None:
        """Register prompt, LLM, and tool in sherma registries."""
        await self.prompt_registry.add(
            RegistryEntry(
                id="weather-system-prompt",
                version="1.0.0",
                instance=Prompt(
                    id="weather-system-prompt",
                    version="1.0.0",
                    instructions=SYSTEM_PROMPT,
                ),
            )
        )

        await self.llm_registry.add(
            RegistryEntry(
                id="openai-gpt-4o-mini",
                version="1.0.0",
                instance=LLM(
                    id="openai-gpt-4o-mini",
                    version="1.0.0",
                    model_name="gpt-4o-mini",
                ),
            )
        )

        sherma_tool = Tool(
            id=get_weather.name,
            version="1.0.0",
            function=get_weather.func,  # type: ignore[union-attr]
        )
        await self.tool_registry.add(
            RegistryEntry(
                id=sherma_tool.id,
                version=sherma_tool.version,
                instance=sherma_tool,
            )
        )

    async def get_graph(self) -> CompiledStateGraph:
        """Build and return the ReAct graph."""
        await self._setup_registries()

        llm_entity = await self.llm_registry.get("openai-gpt-4o-mini")
        prompt_entity = await self.prompt_registry.get("weather-system-prompt")

        llm = ChatOpenAI(
            model=llm_entity.model_name,
            api_key=self.api_key,  # type: ignore[arg-type]
        )
        llm_with_tools = llm.bind_tools([get_weather])

        async def call_model(state: MessagesState) -> dict:
            system_message = {"role": "system", "content": prompt_entity.instructions}
            response = await llm_with_tools.ainvoke(
                [system_message, *state["messages"]]
            )
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode([get_weather]))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        return graph.compile()


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/weather_agent.py <query>")
        sys.exit(1)

    query = sys.argv[1]

    secrets_path = Path(__file__).resolve().parent.parent / "secrets.json"
    if not secrets_path.exists():
        print(
            f"Error: {secrets_path} not found. "
            "Copy secrets.example.json and fill in your API key."
        )
        sys.exit(1)

    secrets = json.loads(secrets_path.read_text())
    api_key = secrets["openai_api_key"]

    agent = WeatherAgent(id="weather-agent", version="1.0.0", api_key=api_key)

    from a2a.types import Message as A2AMessage
    from a2a.types import Part, Role, TextPart

    message = A2AMessage(
        message_id="user-1",
        parts=[Part(root=TextPart(text=query))],
        role=Role.user,
    )

    received = False
    async for event in agent.send_message(message):
        if isinstance(event, A2AMessage):
            for part in event.parts:
                if part.root.kind == "text":
                    print(part.root.text)
                    received = True
    if not received:
        print("No response received.")


if __name__ == "__main__":
    asyncio.run(main())
