from typing import Any

import pytest

from sherma.entities.agent.local import LocalAgent


class MyLocalAgent(LocalAgent):
    async def send_message(self, message: Any) -> Any:
        return {"response": message}

    async def cancel_task(self, task_id: str) -> None:
        pass


@pytest.mark.asyncio
async def test_local_agent():
    agent = MyLocalAgent(id="my-agent", version="1.0.0")
    result = await agent.send_message("hi")
    assert result == {"response": "hi"}
