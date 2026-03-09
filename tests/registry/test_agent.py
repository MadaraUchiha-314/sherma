from typing import Any

import pytest

from sherma.entities.agent.base import Agent
from sherma.registry.agent import AgentRegistry
from sherma.registry.base import RegistryEntry


class MockAgent(Agent):
    async def send_message(self, message: Any) -> Any:
        return message

    async def cancel_task(self, task_id: str) -> None:
        pass


@pytest.mark.asyncio
async def test_agent_registry_local():
    reg = AgentRegistry()
    agent = MockAgent(id="a1", version="1.0.0")
    await reg.add(RegistryEntry(id="a1", version="1.0.0", instance=agent))
    result = await reg.get("a1", "==1.0.0")
    assert result.id == "a1"
