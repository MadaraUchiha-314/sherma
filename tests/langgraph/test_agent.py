"""Unit tests for LangGraphAgent interrupt handling."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from a2a.client.client import ClientEvent
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Interrupt

from sherma.langgraph.agent import LangGraphAgent


class MockLangGraphAgent(LangGraphAgent):
    """Concrete LangGraphAgent with a mock graph."""

    def __init__(self, graph: CompiledStateGraph, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._graph = graph

    async def get_graph(self) -> CompiledStateGraph:
        return self._graph


def _make_message(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> Message:
    return Message(
        message_id="m1",
        parts=[Part(root=TextPart(text=text))],
        role=Role.user,
        task_id=task_id,
        context_id=context_id,
    )


@pytest.mark.asyncio
async def test_send_message_normal_response():
    """LangGraphAgent yields a Message for normal (non-interrupted) responses."""
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.ainvoke = AsyncMock(
        return_value={
            "messages": [AIMessage(content="hello back")],
        }
    )

    agent = MockLangGraphAgent(graph=graph, id="test-agent")
    request = _make_message("hello")

    events: list[ClientEvent | Message] = []
    async for event in agent.send_message(request):
        events.append(event)

    assert len(events) == 1
    msg = events[0]
    assert isinstance(msg, Message)
    assert msg.role == Role.agent
    assert msg.parts[0].root.text == "hello back"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_send_message_single_interrupt():
    """LangGraphAgent yields TaskStatusUpdateEvent(input_required) on interrupt."""
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.ainvoke = AsyncMock(
        return_value={
            "messages": [AIMessage(content="partial")],
            "__interrupt__": (Interrupt(value="What is your name?"),),
        }
    )

    agent = MockLangGraphAgent(graph=graph, id="test-agent")
    request = _make_message("hello", task_id="t1", context_id="ctx1")

    events: list[ClientEvent | Message] = []
    async for event in agent.send_message(request):
        events.append(event)

    assert len(events) == 1
    event = events[0]
    # Should be a ClientEvent tuple
    assert isinstance(event, tuple)
    task, update = event
    assert isinstance(task, Task)
    assert isinstance(update, TaskStatusUpdateEvent)
    assert update.status.state == TaskState.input_required
    assert update.final is False
    # Message should have one part with the interrupt value
    assert update.status.message is not None
    assert len(update.status.message.parts) == 1
    assert update.status.message.parts[0].root.text == "What is your name?"  # type: ignore[union-attr]
    # task_id and context_id should be forwarded
    assert update.task_id == "t1"
    assert update.context_id == "ctx1"


@pytest.mark.asyncio
async def test_send_message_multiple_interrupts():
    """All interrupt values become parts of a single message."""
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.ainvoke = AsyncMock(
        return_value={
            "messages": [],
            "__interrupt__": (
                Interrupt(value="What is your name?"),
                Interrupt(value="What is your age?"),
            ),
        }
    )

    agent = MockLangGraphAgent(graph=graph, id="test-agent")
    request = _make_message("hello", task_id="t1", context_id="ctx1")

    events: list[ClientEvent | Message] = []
    async for event in agent.send_message(request):
        events.append(event)

    assert len(events) == 1
    _task, update = events[0]
    assert isinstance(update, TaskStatusUpdateEvent)
    assert update.status.state == TaskState.input_required
    # Two parts — one per interrupt
    msg = update.status.message
    assert msg is not None
    assert len(msg.parts) == 2
    assert msg.parts[0].root.text == "What is your name?"  # type: ignore[union-attr]
    assert msg.parts[1].root.text == "What is your age?"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_send_message_no_messages():
    """LangGraphAgent yields nothing when graph returns no messages."""
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.ainvoke = AsyncMock(return_value={"messages": []})

    agent = MockLangGraphAgent(graph=graph, id="test-agent")
    request = _make_message("hello")

    events: list[ClientEvent | Message] = []
    async for event in agent.send_message(request):
        events.append(event)

    assert len(events) == 0
