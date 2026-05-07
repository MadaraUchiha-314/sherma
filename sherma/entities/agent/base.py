from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from a2a.client.client import UpdateEvent
from a2a.client.middleware import ClientCallContext
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Message,
    Task,
    TaskIdParams,
)
from pydantic import BaseModel

from sherma.entities.base import EntityBase
from sherma.schema import SCHEMA_INPUT_URI, SCHEMA_OUTPUT_URI, schema_to_extension


class Agent(EntityBase, ABC):
    """Abstract base class for all agents.

    ``input_schema`` and ``output_schema`` may be either a Pydantic
    model class or a raw JSON Schema dict. Validation utilities and
    A2A integration dispatch on the value's type, so both forms are
    treated identically downstream.
    """

    agent_card: AgentCard | None = None
    input_schema: type[BaseModel] | dict[str, Any] | None = None
    output_schema: type[BaseModel] | dict[str, Any] | None = None

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
        """Get the agent's A2A card, with schema extensions auto-injected."""
        if self.agent_card is None:
            return None

        schema_extensions = []
        if self.input_schema is not None:
            schema_extensions.append(
                schema_to_extension(SCHEMA_INPUT_URI, self.input_schema)
            )
        if self.output_schema is not None:
            schema_extensions.append(
                schema_to_extension(SCHEMA_OUTPUT_URI, self.output_schema)
            )

        if not schema_extensions:
            return self.agent_card

        # Build a new capabilities with merged extensions (avoid duplicates)
        existing_caps = self.agent_card.capabilities or AgentCapabilities()
        existing_extensions = list(existing_caps.extensions or [])
        existing_uris = {ext.uri for ext in existing_extensions}
        for ext in schema_extensions:
            if ext.uri not in existing_uris:
                existing_extensions.append(ext)

        new_caps = existing_caps.model_copy(update={"extensions": existing_extensions})
        return self.agent_card.model_copy(update={"capabilities": new_caps})
