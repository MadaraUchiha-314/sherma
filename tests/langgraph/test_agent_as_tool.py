"""Tests for agent_to_langgraph_tool wrapper."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from a2a.types import AgentCapabilities, AgentCard, Message, Part, Role, TextPart
from pydantic import BaseModel

from sherma.entities.agent.base import Agent
from sherma.langgraph.tools import agent_to_langgraph_tool


class _StubAgent(Agent):
    """Minimal agent stub for testing."""

    _response_text: str = "Hello from agent"

    def send_message(
        self,
        request: Message,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            yield Message(
                message_id="resp-1",
                role=Role.agent,
                parts=[Part(root=TextPart(text=self._response_text))],
            )

        return _gen()

    async def cancel_task(self, request: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class WeatherInput(BaseModel):
    city: str
    units: str = "celsius"


@pytest.mark.asyncio
async def test_agent_to_tool_no_input_schema():
    agent = _StubAgent(id="test-agent", version="1.0")
    tool = agent_to_langgraph_tool(agent)

    assert tool.name == "test-agent"
    assert "Invoke agent: test-agent" in tool.description

    # Tool schema should have only 'request'
    schema = tool.args_schema
    assert schema is not None
    fields = schema.model_fields
    assert "request" in fields
    assert "agent_input" not in fields


@pytest.mark.asyncio
async def test_agent_to_tool_with_input_schema():
    agent = _StubAgent(
        id="weather-agent",
        version="1.0",
        input_schema=WeatherInput,
    )
    tool = agent_to_langgraph_tool(agent)

    assert tool.name == "weather-agent"

    # Tool schema should have both 'request' and 'agent_input'
    schema = tool.args_schema
    assert schema is not None
    fields = schema.model_fields
    assert "request" in fields
    assert "agent_input" in fields


@pytest.mark.asyncio
async def test_agent_to_tool_uses_agent_card_description():
    agent = _StubAgent(
        id="my-agent",
        version="1.0",
        agent_card=AgentCard(
            name="My Agent",
            url="http://localhost",
            version="1.0",
            description="A helpful agent",
            capabilities=AgentCapabilities(),
            skills=[],
            defaultInputModes=["text/plain"],
            defaultOutputModes=["text/plain"],
        ),
    )
    tool = agent_to_langgraph_tool(agent)
    assert tool.description == "A helpful agent"


@pytest.mark.asyncio
async def test_agent_to_tool_invoke_no_schema():
    agent = _StubAgent(id="echo-agent", version="1.0")
    tool = agent_to_langgraph_tool(agent)

    result = await tool.ainvoke({"request": "Hello"})
    assert result == "Hello from agent"


@pytest.mark.asyncio
async def test_agent_to_tool_invoke_with_schema():
    agent = _StubAgent(
        id="weather-agent",
        version="1.0",
        input_schema=WeatherInput,
    )
    tool = agent_to_langgraph_tool(agent)

    result = await tool.ainvoke(
        {
            "request": "What is the weather?",
            "agent_input": {"city": "London", "units": "celsius"},
        }
    )
    assert result == "Hello from agent"


@pytest.mark.asyncio
async def test_agent_to_tool_message_has_data_part():
    """Verify that when agent_input is provided, the A2A message includes a DataPart."""
    received_messages: list[Message] = []

    class CapturingAgent(_StubAgent):
        def send_message(
            self,
            request: Message,
            **kwargs: Any,
        ) -> AsyncIterator[Any]:
            received_messages.append(request)
            return super().send_message(request, **kwargs)

    agent = CapturingAgent(
        id="capturing-agent",
        version="1.0",
        input_schema=WeatherInput,
    )
    tool = agent_to_langgraph_tool(agent)

    await tool.ainvoke(
        {
            "request": "Get weather",
            "agent_input": {"city": "Paris"},
        }
    )

    assert len(received_messages) == 1
    msg = received_messages[0]

    # Should have 2 parts: TextPart + DataPart
    assert len(msg.parts) == 2
    assert msg.parts[0].root.kind == "text"
    assert msg.parts[0].root.text == "Get weather"
    assert msg.parts[1].root.kind == "data"
    assert msg.parts[1].root.data["city"] == "Paris"
    assert msg.parts[1].root.metadata["agent_input"] is True


@pytest.mark.asyncio
async def test_agent_to_tool_empty_response():
    """Agent returning no results should produce empty string."""

    class EmptyAgent(_StubAgent):
        def send_message(
            self,
            request: Message,
            **kwargs: Any,
        ) -> AsyncIterator[Any]:
            async def _gen() -> AsyncIterator[Any]:
                return
                yield

            return _gen()

    agent = EmptyAgent(id="empty-agent", version="1.0")
    tool = agent_to_langgraph_tool(agent)

    result = await tool.ainvoke({"request": "Hello"})
    assert result == ""
