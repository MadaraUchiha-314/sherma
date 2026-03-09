from collections.abc import AsyncIterator
from typing import Any

import pytest
from a2a.client.client import UpdateEvent
from a2a.client.middleware import ClientCallContext
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskIdParams,
    TaskState,
    TaskStatus,
    TextPart,
)

from sherma.entities.agent.base import Agent


def _make_message(text: str, message_id: str = "m1") -> Message:
    return Message(
        message_id=message_id,
        parts=[Part(root=TextPart(text=text))],
        role=Role.user,
    )


class ConcreteAgent(Agent):
    async def send_message(
        self,
        request: Message,
        *,
        context: ClientCallContext | None = None,
        request_metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncIterator[UpdateEvent | Message | Task]:
        yield Message(
            message_id="resp-1",
            parts=[Part(root=TextPart(text=f"echo: {request.parts[0].root.text}"))],
            role=Role.agent,
        )

    async def cancel_task(
        self,
        request: TaskIdParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task:
        return Task(
            id=request.id,
            context_id="ctx-1",
            status=TaskStatus(state=TaskState.canceled),
        )


def test_agent_creation():
    a = ConcreteAgent(id="test-agent", version="1.0.0")
    assert a.id == "test-agent"
    assert a.agent_card is None


@pytest.mark.asyncio
async def test_agent_send_message():
    a = ConcreteAgent(id="test-agent")
    msg = _make_message("hello")
    results = [event async for event in a.send_message(msg)]
    assert len(results) == 1
    assert isinstance(results[0], Message)


@pytest.mark.asyncio
async def test_agent_get_card_default():
    a = ConcreteAgent(id="test-agent")
    card = await a.get_card()
    assert card is None
