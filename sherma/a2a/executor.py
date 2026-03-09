"""A2A AgentExecutor implementation for sherma agents."""

from __future__ import annotations

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskStatusUpdateEvent,
)
from a2a.utils.task import new_task

from sherma.entities.agent.base import Agent
from sherma.logging import get_logger

logger = get_logger(__name__)


class ShermaAgentExecutor(AgentExecutor):
    """An A2A AgentExecutor that delegates to a sherma Agent.

    Bridges the A2A server protocol to sherma's Agent interface by
    forwarding ``execute`` and ``cancel`` calls.
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute a message through the agent."""
        # Get or create task
        task: Task | None = context.current_task
        if task is None:
            message = context.message
            if message is None:
                logger.warning("Execute called without a message")
                return
            task = new_task(message)
        context.current_task = task

        logger.info(
            "Executing message for task=%s context=%s",
            task.id,
            task.context_id,
        )

        # Create TaskUpdater and signal work has begun
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        await task_updater.start_work()

        # Ensure message has task_id and context_id
        message = context.message
        if message is None:
            logger.warning("Execute called without a message")
            return
        message.task_id = task.id
        message.context_id = task.context_id

        # Call agent and process responses
        has_events = False
        async for event in self.agent.send_message(message):
            has_events = True
            if isinstance(event, Message):
                await task_updater.complete(message=event)
            elif isinstance(event, Task):
                logger.debug(
                    "Received initial task event for task=%s",
                    event.id,
                )
            elif isinstance(event, TaskArtifactUpdateEvent):
                artifact = event.artifact
                await task_updater.add_artifact(
                    parts=artifact.parts,
                    artifact_id=artifact.artifact_id,
                    name=artifact.name,
                    metadata=artifact.metadata,
                    append=event.append,
                    last_chunk=event.last_chunk,
                )
            elif isinstance(event, TaskStatusUpdateEvent):
                await task_updater.update_status(
                    state=event.status.state,
                    message=event.status.message,
                    final=event.final,
                )

        if not has_events:
            await task_updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel a running task."""
        task: Task | None = context.current_task
        task_id = task.id if task else context.task_id
        if task_id is None:
            logger.warning("Cancel called without a task_id")
            return

        logger.info("Cancelling task %s", task_id)

        context_id = task.context_id if task else (context.context_id or task_id)
        task_updater = TaskUpdater(event_queue, task_id, context_id)

        params = TaskIdParams(id=task_id)
        await self.agent.cancel_task(params)
        await task_updater.cancel()
