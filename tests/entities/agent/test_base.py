from collections.abc import AsyncIterator
from typing import Any

import pytest
from a2a.client.client import UpdateEvent
from a2a.client.middleware import ClientCallContext
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    Message,
    Part,
    Role,
    Task,
    TaskIdParams,
    TaskState,
    TaskStatus,
    TextPart,
)
from pydantic import BaseModel

from sherma.entities.agent.base import Agent
from sherma.schema import SCHEMA_INPUT_URI, SCHEMA_OUTPUT_URI


def _make_card(**kwargs: Any) -> AgentCard:
    defaults: dict[str, Any] = {
        "name": "test",
        "url": "http://localhost",
        "version": "1.0",
        "skills": [],
        "capabilities": AgentCapabilities(),
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "description": "A test agent",
    }
    defaults.update(kwargs)
    return AgentCard(**defaults)


def _make_message(text: str, message_id: str = "m1") -> Message:
    return Message(
        message_id=message_id,
        parts=[Part(root=TextPart(text=text))],
        role=Role.user,
    )


class ConcreteAgent(Agent):
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
            parts=[Part(root=TextPart(text=f"echo: {request.parts[0].root.text}"))],
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


def test_agent_creation():
    a = ConcreteAgent(id="test-agent", version="1.0.0")
    assert a.id == "test-agent"
    assert a.agent_card is None


@pytest.mark.asyncio
async def test_agent_send_message():
    a = ConcreteAgent(id="test-agent")
    msg = _make_message("hello")
    results = [event async for event in a.send_message(msg)]
    assert len(results) == 1
    assert isinstance(results[0], Message)


@pytest.mark.asyncio
async def test_agent_get_card_default():
    a = ConcreteAgent(id="test-agent")
    card = await a.get_card()
    assert card is None


class InputModel(BaseModel):
    name: str
    value: int


class OutputModel(BaseModel):
    result: str


@pytest.mark.asyncio
async def test_get_card_injects_schema_extensions():
    card = _make_card()
    a = ConcreteAgent(
        id="schema-agent",
        agent_card=card,
        input_schema=InputModel,
        output_schema=OutputModel,
    )
    result = await a.get_card()
    assert result is not None
    assert result.capabilities is not None
    extensions = result.capabilities.extensions
    assert extensions is not None
    assert len(extensions) == 2

    uris = {ext.uri for ext in extensions}
    assert SCHEMA_INPUT_URI in uris
    assert SCHEMA_OUTPUT_URI in uris

    # Verify schemas are present in params
    for ext in extensions:
        assert ext.params is not None
        assert "properties" in ext.params


@pytest.mark.asyncio
async def test_get_card_no_schema_returns_card_unchanged():
    card = _make_card()
    a = ConcreteAgent(id="no-schema-agent", agent_card=card)
    result = await a.get_card()
    assert result is card


@pytest.mark.asyncio
async def test_get_card_does_not_duplicate_extensions():
    existing_ext = AgentExtension(uri=SCHEMA_INPUT_URI, params={"custom": True})
    card = _make_card(capabilities=AgentCapabilities(extensions=[existing_ext]))
    a = ConcreteAgent(
        id="dup-agent",
        agent_card=card,
        input_schema=InputModel,
    )
    result = await a.get_card()
    assert result is not None
    extensions = result.capabilities.extensions
    assert extensions is not None
    input_exts = [ext for ext in extensions if ext.uri == SCHEMA_INPUT_URI]
    assert len(input_exts) == 1
    # Should keep the existing one, not inject a new one
    assert input_exts[0].params == {"custom": True}


@pytest.mark.asyncio
async def test_get_card_preserves_existing_capabilities():
    card = _make_card(capabilities=AgentCapabilities(streaming=True))
    a = ConcreteAgent(
        id="caps-agent",
        agent_card=card,
        input_schema=InputModel,
    )
    result = await a.get_card()
    assert result is not None
    assert result.capabilities is not None
    assert result.capabilities.streaming is True
    assert result.capabilities.extensions is not None
    assert len(result.capabilities.extensions) == 1


JSON_OUTPUT_SCHEMA: dict = {
    "title": "Result",
    "type": "object",
    "required": ["result"],
    "properties": {"result": {"type": "string"}},
}


@pytest.mark.asyncio
async def test_get_card_supports_json_schema_dict_output():
    """A dict-typed output_schema is published as an A2A extension."""
    card = _make_card()
    a = ConcreteAgent(
        id="json-schema-agent",
        agent_card=card,
        output_schema=JSON_OUTPUT_SCHEMA,
    )
    result = await a.get_card()
    assert result is not None
    assert result.capabilities is not None
    extensions = result.capabilities.extensions
    assert extensions is not None
    output_exts = [ext for ext in extensions if ext.uri == SCHEMA_OUTPUT_URI]
    assert len(output_exts) == 1
    assert output_exts[0].params == JSON_OUTPUT_SCHEMA


@pytest.mark.asyncio
async def test_get_card_mixes_pydantic_and_json_schema():
    """Pydantic-typed input + dict-typed output produce two distinct
    extensions."""
    card = _make_card()
    a = ConcreteAgent(
        id="mixed-agent",
        agent_card=card,
        input_schema=InputModel,
        output_schema=JSON_OUTPUT_SCHEMA,
    )
    result = await a.get_card()
    assert result is not None
    extensions = result.capabilities.extensions  # type: ignore[union-attr]
    assert extensions is not None
    uris = {ext.uri for ext in extensions}
    assert SCHEMA_INPUT_URI in uris
    assert SCHEMA_OUTPUT_URI in uris
