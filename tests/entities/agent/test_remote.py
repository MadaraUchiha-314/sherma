from unittest.mock import AsyncMock

import pytest

from sherma.entities.agent.remote import RemoteAgent


@pytest.mark.asyncio
async def test_remote_agent_send_message():
    mock_client = AsyncMock()
    mock_client.send_message.return_value = {"ok": True}
    agent = RemoteAgent(id="remote", version="1.0.0", client=mock_client)
    result = await agent.send_message("hello")
    assert result == {"ok": True}
    mock_client.send_message.assert_called_once_with("hello")


@pytest.mark.asyncio
async def test_remote_agent_cancel_task():
    mock_client = AsyncMock()
    agent = RemoteAgent(id="remote", client=mock_client)
    await agent.cancel_task("task-123")
    mock_client.cancel_task.assert_called_once_with("task-123")


@pytest.mark.asyncio
async def test_remote_agent_no_client_raises():
    agent = RemoteAgent(id="remote")
    with pytest.raises(RuntimeError, match="requires an A2A client"):
        await agent.send_message("hi")


@pytest.mark.asyncio
async def test_remote_agent_get_card_from_client():
    mock_client = AsyncMock()
    mock_client.get_card.return_value = {"name": "remote-agent"}
    agent = RemoteAgent(id="remote", client=mock_client)
    card = await agent.get_card()
    assert card == {"name": "remote-agent"}
