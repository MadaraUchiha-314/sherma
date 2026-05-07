"""A2A AgentExecutor implementation for sherma agents."""

from __future__ import annotations

import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils.task import new_task

from sherma.entities.agent.base import Agent
from sherma.logging import get_logger
from sherma.schema import validate_against_schema

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

        # Validate incoming DataParts against input_schema
        if self.agent.input_schema is not None:
            for part in message.parts:
                if (
                    isinstance(part.root, DataPart)
                    and part.root.metadata is not None
                    and part.root.metadata.get("agent_input") is True
                ):
                    validate_against_schema(part.root.data, self.agent.input_schema)

        # Call agent and process responses
        try:
            has_events = False
            async for event in self.agent.send_message(message):
                has_events = True
                if isinstance(event, Message):
                    # Validate outgoing DataParts against output_schema
                    if self.agent.output_schema is not None:
                        for part in event.parts:
                            if (
                                isinstance(part.root, DataPart)
                                and part.root.metadata is not None
                                and part.root.metadata.get("agent_output") is True
                            ):
                                validate_against_schema(
                                    part.root.data, self.agent.output_schema
                                )
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
        except Exception as exc:
            logger.error("Agent execution failed for task=%s: %s", task.id, str(exc))
            error_message = Message(
                message_id=str(uuid.uuid4()),
                role=Role.agent,
                parts=[Part(root=TextPart(text=f"Agent execution failed: {exc}"))],
            )
            await task_updater.failed(message=error_message)

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
