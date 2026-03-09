from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from sherma.entities.base import EntityBase

if TYPE_CHECKING:
    from a2a.types import AgentCard


class Agent(EntityBase, ABC):
    """Abstract base class for all agents."""

    agent_card: Any | None = None  # AgentCard when a2a-sdk is available

    @abstractmethod
    async def send_message(self, message: Any) -> Any:
        """Send a message to the agent."""
        ...

    @abstractmethod
    async def cancel_task(self, task_id: str) -> None:
        """Cancel a running task."""
        ...

    async def get_card(self) -> "AgentCard | Any | None":
        """Get the agent's A2A card."""
        return self.agent_card
