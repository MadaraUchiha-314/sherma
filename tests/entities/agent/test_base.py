from typing import Any

import pytest

from sherma.entities.agent.base import Agent


class ConcreteAgent(Agent):
    async def send_message(self, message: Any) -> Any:
        return f"echo: {message}"

    async def cancel_task(self, task_id: str) -> None:
        pass


def test_agent_creation():
    a = ConcreteAgent(id="test-agent", version="1.0.0")
    assert a.id == "test-agent"
    assert a.agent_card is None


@pytest.mark.asyncio
async def test_agent_send_message():
    a = ConcreteAgent(id="test-agent")
    result = await a.send_message("hello")
    assert result == "echo: hello"


@pytest.mark.asyncio
async def test_agent_get_card_default():
    a = ConcreteAgent(id="test-agent")
    card = await a.get_card()
    assert card is None
