from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from a2a.client import Client, ClientConfig, ClientFactory
from a2a.client.client import ClientEvent
from a2a.client.middleware import ClientCallContext
from a2a.types import (
    AgentCard,
    Message,
    Task,
    TaskIdParams,
)
from pydantic import ConfigDict

from sherma.entities.agent.base import Agent
from sherma.http import get_http_client
from sherma.logging import get_logger

logger = get_logger(__name__)


class RemoteAgent(Agent):
    """An agent that delegates to a remote A2A-compatible agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str | None = None
    client: Client | None = None

    async def _get_or_create_client(self) -> Client:
        """Return the client, creating one from agent_card or url if needed."""
        if self.client is not None:
            return self.client

        httpx_client = await get_http_client()
        config = ClientConfig(httpx_client=httpx_client)

        if self.agent_card is not None:
            self.client = await ClientFactory.connect(
                self.agent_card, client_config=config
            )
            return self.client

        if self.url is not None:
            self.client = await ClientFactory.connect(self.url, client_config=config)
            self.agent_card = await self.client.get_card()
            return self.client

        msg = "RemoteAgent requires either a client, agent_card, or url"
        raise RuntimeError(msg)

    async def send_message(
        self,
        request: Message,
        *,
        context: ClientCallContext | None = None,
        request_metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncIterator[ClientEvent | Message]:
        """Send a message via the A2A client and yield response events."""
        client = await self._get_or_create_client()
        async for event in client.send_message(
            request,
            context=context,
            request_metadata=request_metadata,
            extensions=extensions,
        ):
            yield event

    async def cancel_task(
        self,
        request: TaskIdParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task:
        """Cancel a task via the A2A client."""
        client = await self._get_or_create_client()
        return await client.cancel_task(request, context=context, extensions=extensions)

    async def get_card(self) -> AgentCard | None:
        """Get the agent card, fetching from remote if needed."""
        if self.agent_card is not None:
            return self.agent_card
        if self.url is not None or self.client is not None:
            client = await self._get_or_create_client()
            self.agent_card = await client.get_card()
            return self.agent_card
        return None
