from collections.abc import AsyncIterator
from typing import Any

import pytest
from a2a.client.client import ClientEvent
from a2a.client.middleware import ClientCallContext
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskIdParams,
    TaskState,
    TaskStatus,
    TextPart,
)

from sherma.a2a.executor import ShermaAgentExecutor
from sherma.entities.agent.base import Agent


def _make_message(text: str, message_id: str = "m1") -> Message:
    return Message(
        message_id=message_id,
        parts=[Part(root=TextPart(text=text))],
        role=Role.user,
    )


class EchoAgent(Agent):
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
            parts=[Part(root=TextPart(text="echo"))],
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


def test_executor_is_a2a_agent_executor():
    agent = EchoAgent(id="echo")
    executor = ShermaAgentExecutor(agent)
    assert isinstance(executor, AgentExecutor)


@pytest.mark.asyncio
async def test_executor_execute():
    agent = EchoAgent(id="echo")
    executor = ShermaAgentExecutor(agent)

    message = _make_message("hello")
    request = MessageSendParams(message=message)
    context = RequestContext(request=request, task_id="t1")
    event_queue = EventQueue()

    await executor.execute(context, event_queue)

    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, Message)
    assert event.role == Role.agent


@pytest.mark.asyncio
async def test_executor_cancel():
    agent = EchoAgent(id="echo")
    executor = ShermaAgentExecutor(agent)

    context = RequestContext(task_id="task-1")
    event_queue = EventQueue()

    await executor.cancel(context, event_queue)

    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, Task)
    assert event.id == "task-1"
    assert event.status.state == TaskState.canceled


@pytest.mark.asyncio
async def test_executor_cancel_without_task_id():
    agent = EchoAgent(id="echo")
    executor = ShermaAgentExecutor(agent)

    context = RequestContext()
    event_queue = EventQueue()

    await executor.cancel(context, event_queue)

    import asyncio

    with pytest.raises(asyncio.QueueEmpty):
        await event_queue.dequeue_event(no_wait=True)
