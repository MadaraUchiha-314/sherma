from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sherma.entities.agent.base import Agent
from sherma.logging import get_logger

if TYPE_CHECKING:
    from a2a.types import AgentCard

logger = get_logger(__name__)


class RemoteAgent(Agent):
    """An agent that delegates to a remote A2A-compatible agent."""

    client: Any = None  # A2AClient when a2a-sdk is available

    def _require_client(self) -> Any:
        """Return the client, raising if not set."""
        if self.client is None:
            msg = "RemoteAgent requires an A2A client"
            raise RuntimeError(msg)
        return self.client

    async def send_message(self, message: Any) -> Any:
        """Send a message via the A2A client."""
        return await self._require_client().send_message(message)

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a task via the A2A client."""
        await self._require_client().cancel_task(task_id)

    async def get_card(self) -> AgentCard | Any | None:
        """Get the agent card, fetching from remote if needed."""
        if self.agent_card is not None:
            return self.agent_card
        if self.client is not None:
            card = await self.client.get_card()
            self.agent_card = card
            return card
        return None
