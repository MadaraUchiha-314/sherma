from typing import Any

import pytest

from sherma.a2a.executor import ShermaAgentExecutor
from sherma.entities.agent.base import Agent


class EchoAgent(Agent):
    async def send_message(self, message: Any) -> Any:
        return {"echo": message}

    async def cancel_task(self, task_id: str) -> None:
        pass


@pytest.mark.asyncio
async def test_executor_execute():
    agent = EchoAgent(id="echo")
    executor = ShermaAgentExecutor(agent)
    result = await executor.execute("hello", task_id="t1")
    assert result == {"echo": "hello"}


@pytest.mark.asyncio
async def test_executor_cancel():
    agent = EchoAgent(id="echo")
    executor = ShermaAgentExecutor(agent)
    await executor.cancel("task-1")
