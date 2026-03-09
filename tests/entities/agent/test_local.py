from collections.abc import AsyncIterator
from typing import Any

import pytest
from a2a.client.client import ClientEvent
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

from sherma.entities.agent.local import LocalAgent


class MyLocalAgent(LocalAgent):
    async def send_message(
        self,
        request: Message,
        *,
        context: ClientCallContext | None = None,
        request_metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncIterator[ClientEvent | Message]:
        yield Message(
            message_id="resp-1",
            parts=[Part(root=TextPart(text="response"))],
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


@pytest.mark.asyncio
async def test_local_agent():
    agent = MyLocalAgent(id="my-agent", version="1.0.0")
    msg = Message(
        message_id="m1",
        parts=[Part(root=TextPart(text="hi"))],
        role=Role.user,
    )
    results = [event async for event in agent.send_message(msg)]
    assert len(results) == 1
    assert isinstance(results[0], Message)
