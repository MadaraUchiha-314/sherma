"""Integration test: programmatic LangGraphAgent with ReAct graph.

Mirrors examples/weather_agent/main.py with mocked LLM.
"""

from __future__ import annotations

from typing import Any

import pytest
from a2a.types import Message as A2AMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import ConfigDict

from sherma.langgraph.agent import LangGraphAgent
from tests.integration.conftest import FakeChatModel, collect_events, make_a2a_message
from tests.integration.fake_tools import get_weather

SYSTEM_PROMPT = (
    "You are a helpful weather assistant. "
    "Use the get_weather tool to look up current weather."
)


class FakeWeatherAgent(LangGraphAgent):
    """Programmatic ReAct agent with a FakeChatModel."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    fake_model: Any = None

    async def get_graph(self) -> CompiledStateGraph:
        llm_with_tools = self.fake_model.bind_tools([get_weather])

        async def call_model(state: MessagesState) -> dict:
            system_message = {"role": "system", "content": SYSTEM_PROMPT}
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
        return graph.compile(checkpointer=MemorySaver())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basic_weather_query():
    """LLM calls tool, gets result, produces final text response."""
    fake_model = FakeChatModel(
        responses=[
            # First call: LLM decides to call get_weather
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"city": "Tokyo"},
                    }
                ],
            ),
            # Second call: LLM produces final text answer
            AIMessage(content="Tokyo is 20C with clear skies and 5km/h wind."),
        ],
    )

    agent = FakeWeatherAgent(
        id="weather-agent",
        version="1.0.0",
        fake_model=fake_model,
    )
    message = make_a2a_message("What is the weather in Tokyo?")
    events = await collect_events(agent, message)

    # Should get an A2A Message back with the weather info
    assert len(events) >= 1
    last = events[-1]
    assert isinstance(last, A2AMessage)
    text = last.parts[0].root.text
    assert "20C" in text or "Tokyo" in text

    # Model was called twice: once for tool call, once for final answer
    assert fake_model.call_count == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_tool_call():
    """LLM returns text directly without invoking any tool."""
    fake_model = FakeChatModel(
        responses=[
            AIMessage(content="I don't need a tool for that. Hello!"),
        ],
    )

    agent = FakeWeatherAgent(
        id="weather-agent",
        version="1.0.0",
        fake_model=fake_model,
    )
    message = make_a2a_message("Hello!")
    events = await collect_events(agent, message)

    assert len(events) >= 1
    last = events[-1]
    assert isinstance(last, A2AMessage)
    assert "Hello" in last.parts[0].root.text
    assert fake_model.call_count == 1
