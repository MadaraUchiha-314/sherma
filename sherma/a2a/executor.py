"""A2A AgentExecutor implementation for sherma agents."""

from __future__ import annotations

from typing import Any

from sherma.entities.agent.base import Agent
from sherma.logging import get_logger

logger = get_logger(__name__)


class ShermaAgentExecutor:
    """An A2A AgentExecutor that delegates to a sherma Agent.

    Manages task lifecycle using TaskUpdater when a2a-sdk is available.
    Falls back to simple invocation otherwise.
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    async def execute(
        self,
        message: Any,
        *,
        task_id: str | None = None,
        context_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a message through the agent."""
        logger.info("Executing message for task=%s context=%s", task_id, context_id)
        return await self.agent.send_message(message)

    async def cancel(self, task_id: str, **kwargs: Any) -> None:
        """Cancel a running task."""
        logger.info("Cancelling task %s", task_id)
        await self.agent.cancel_task(task_id)
