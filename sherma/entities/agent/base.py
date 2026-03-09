from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from a2a.client.client import UpdateEvent
from a2a.client.middleware import ClientCallContext
from a2a.types import (
    AgentCard,
    Message,
    Task,
    TaskIdParams,
)

from sherma.entities.base import EntityBase


class Agent(EntityBase, ABC):
    """Abstract base class for all agents."""

    agent_card: AgentCard | None = None

    @abstractmethod
    def send_message(
        self,
        request: Message,
        *,
        context: ClientCallContext | None = None,
        request_metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncIterator[UpdateEvent | Message | Task]:
        """Send a message to the agent and yield response events."""
        ...

    @abstractmethod
    async def cancel_task(
        self,
        request: TaskIdParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task:
        """Cancel a running task."""
        ...

    async def get_card(self) -> AgentCard | None:
        """Get the agent's A2A card."""
        return self.agent_card
