"""A2A AgentExecutor implementation for sherma agents."""

from __future__ import annotations

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, TaskIdParams

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
        logger.info(
            "Executing message for task=%s context=%s",
            context.task_id,
            context.context_id,
        )
        message = context.message
        if message is None:
            logger.warning("Execute called without a message")
            return
        async for event in self.agent.send_message(message):
            if isinstance(event, Message):
                await event_queue.enqueue_event(event)
            else:
                # ClientEvent is tuple[Task, UpdateEvent]
                task, update = event
                if update is not None:
                    await event_queue.enqueue_event(update)
                await event_queue.enqueue_event(task)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel a running task."""
        logger.info("Cancelling task %s", context.task_id)
        params = TaskIdParams(id=context.task_id) if context.task_id else None
        if params is None:
            logger.warning("Cancel called without a task_id")
            return
        task = await self.agent.cancel_task(params)
        await event_queue.enqueue_event(task)
