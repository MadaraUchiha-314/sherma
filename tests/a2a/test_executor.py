from collections.abc import AsyncIterator
from typing import Any

import pytest
from a2a.client.client import UpdateEvent
from a2a.client.middleware import ClientCallContext
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
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
    ) -> AsyncIterator[UpdateEvent | Message | Task]:
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


class TaskResponseAgent(Agent):
    """Agent that returns UpdateEvents directly."""

    async def send_message(
        self,
        request: Message,
        *,
        context: ClientCallContext | None = None,
        request_metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncIterator[UpdateEvent | Message | Task]:
        task = Task(
            id="task-resp",
            context_id="ctx-resp",
            status=TaskStatus(
                state=TaskState.completed,
                message=Message(
                    message_id="status-msg",
                    parts=[Part(root=TextPart(text="done"))],
                    role=Role.agent,
                ),
            ),
            artifacts=[
                Artifact(
                    artifact_id="art-1",
                    name="result",
                    parts=[Part(root=TextPart(text="artifact content"))],
                )
            ],
        )
        # Yield artifact update event
        yield TaskArtifactUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            artifact=task.artifacts[0],
        )
        # Yield status update event
        yield TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=task.status,
            final=True,
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

    # First event: TaskStatusUpdateEvent for "working"
    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.status.state == TaskState.working

    # Second event: TaskStatusUpdateEvent for "completed" with the agent message
    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.status.state == TaskState.completed
    assert event.final is True
    assert event.status.message is not None
    assert event.status.message.role == Role.agent


@pytest.mark.asyncio
async def test_executor_execute_creates_task():
    """Executor creates a Task on context when none exists."""
    agent = EchoAgent(id="echo")
    executor = ShermaAgentExecutor(agent)

    message = _make_message("hello")
    request = MessageSendParams(message=message)
    context = RequestContext(request=request)
    event_queue = EventQueue()

    assert context.current_task is None
    await executor.execute(context, event_queue)
    assert context.current_task is not None
    assert context.current_task.status.state == TaskState.submitted


@pytest.mark.asyncio
async def test_executor_execute_sets_message_ids():
    """Executor sets task_id and context_id on the message."""
    agent = EchoAgent(id="echo")
    executor = ShermaAgentExecutor(agent)

    message = _make_message("hello")
    assert message.task_id is None
    assert message.context_id is None

    request = MessageSendParams(message=message)
    context = RequestContext(request=request)
    event_queue = EventQueue()

    await executor.execute(context, event_queue)

    # Message should now have task_id and context_id set
    assert message.task_id is not None
    assert message.context_id is not None


@pytest.mark.asyncio
async def test_executor_execute_task_response():
    """Executor handles UpdateEvent responses with artifacts."""
    agent = TaskResponseAgent(id="task-agent")
    executor = ShermaAgentExecutor(agent)

    message = _make_message("hello")
    request = MessageSendParams(message=message)
    context = RequestContext(request=request, task_id="t1")
    event_queue = EventQueue()

    await executor.execute(context, event_queue)

    # First: working status
    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.status.state == TaskState.working

    # Second: artifact update
    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, TaskArtifactUpdateEvent)
    assert event.artifact.artifact_id == "art-1"
    assert event.artifact.name == "result"

    # Third: completed status from task response
    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.status.state == TaskState.completed


@pytest.mark.asyncio
async def test_executor_cancel():
    agent = EchoAgent(id="echo")
    executor = ShermaAgentExecutor(agent)

    context = RequestContext(task_id="task-1")
    event_queue = EventQueue()

    await executor.cancel(context, event_queue)

    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.status.state == TaskState.canceled
    assert event.final is True


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


class StreamingArtifactAgent(Agent):
    """Agent that streams artifact chunks with append/last_chunk."""

    async def send_message(
        self,
        request: Message,
        *,
        context: ClientCallContext | None = None,
        request_metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncIterator[UpdateEvent | Message | Task]:
        task = Task(
            id="task-stream",
            context_id="ctx-stream",
            status=TaskStatus(state=TaskState.working),
        )
        # First chunk
        yield TaskArtifactUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            artifact=Artifact(
                artifact_id="art-stream",
                parts=[Part(root=TextPart(text="chunk-1"))],
            ),
            append=False,
            last_chunk=False,
        )
        # Second chunk (append)
        yield TaskArtifactUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            artifact=Artifact(
                artifact_id="art-stream",
                parts=[Part(root=TextPart(text="chunk-2"))],
            ),
            append=True,
            last_chunk=True,
        )
        # Final status
        yield TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=TaskStatus(state=TaskState.completed),
            final=True,
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
async def test_executor_execute_streaming_artifacts():
    """Executor forwards append/last_chunk from TaskArtifactUpdateEvent."""
    agent = StreamingArtifactAgent(id="stream-agent")
    executor = ShermaAgentExecutor(agent)

    message = _make_message("hello")
    request = MessageSendParams(message=message)
    context = RequestContext(request=request, task_id="t1")
    event_queue = EventQueue()

    await executor.execute(context, event_queue)

    # First: working status from start_work()
    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.status.state == TaskState.working

    # Second: first artifact chunk (append=False, last_chunk=False)
    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, TaskArtifactUpdateEvent)
    assert event.artifact.artifact_id == "art-stream"
    assert event.append is False
    assert event.last_chunk is False

    # Third: second artifact chunk (append=True, last_chunk=True)
    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, TaskArtifactUpdateEvent)
    assert event.artifact.artifact_id == "art-stream"
    assert event.append is True
    assert event.last_chunk is True

    # Fourth: completed status (final=True)
    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.status.state == TaskState.completed
    assert event.final is True


class NullUpdateAgent(Agent):
    """Agent that yields a Task — initial task creation event."""

    async def send_message(
        self,
        request: Message,
        *,
        context: ClientCallContext | None = None,
        request_metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncIterator[UpdateEvent | Message | Task]:
        task = Task(
            id="task-null",
            context_id="ctx-null",
            status=TaskStatus(state=TaskState.working),
        )
        yield task

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
async def test_executor_execute_null_update_event():
    """Executor handles Task events without emitting extra events."""
    agent = NullUpdateAgent(id="null-agent")
    executor = ShermaAgentExecutor(agent)

    message = _make_message("hello")
    request = MessageSendParams(message=message)
    context = RequestContext(request=request, task_id="t1")
    event_queue = EventQueue()

    await executor.execute(context, event_queue)

    # First: working status from start_work()
    event = await event_queue.dequeue_event(no_wait=True)
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.status.state == TaskState.working

    # No more events — the (task, None) event should not produce anything
    import asyncio

    with pytest.raises(asyncio.QueueEmpty):
        await event_queue.dequeue_event(no_wait=True)
