from unittest.mock import AsyncMock

import pytest
from a2a.client import Client
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskIdParams,
    TaskState,
    TaskStatus,
    TextPart,
)

from sherma.entities.agent.remote import RemoteAgent


def _make_message(text: str, message_id: str = "m1") -> Message:
    return Message(
        message_id=message_id,
        parts=[Part(root=TextPart(text=text))],
        role=Role.user,
    )


def _make_task(task_id: str, context_id: str = "ctx-1") -> Task:
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=TaskState.completed),
    )


def _mock_client() -> AsyncMock:
    return AsyncMock(spec=Client)


@pytest.mark.asyncio
async def test_remote_agent_send_message():
    mock_client = _mock_client()
    response_msg = Message(
        message_id="resp-1",
        parts=[Part(root=TextPart(text="ok"))],
        role=Role.agent,
    )

    async def mock_send_message(request, **kwargs):
        yield response_msg

    mock_client.send_message = mock_send_message
    agent = RemoteAgent(id="remote", version="1.0.0", client=mock_client)
    results = [event async for event in agent.send_message(_make_message("hello"))]
    assert len(results) == 1
    assert isinstance(results[0], Message)
    assert results[0].message_id == "resp-1"


@pytest.mark.asyncio
async def test_remote_agent_cancel_task():
    mock_client = _mock_client()
    expected_task = _make_task("task-123")
    mock_client.cancel_task.return_value = expected_task
    agent = RemoteAgent(id="remote", client=mock_client)
    params = TaskIdParams(id="task-123")
    result = await agent.cancel_task(params)
    assert isinstance(result, Task)
    assert result.id == "task-123"


@pytest.mark.asyncio
async def test_remote_agent_no_client_or_url_raises():
    agent = RemoteAgent(id="remote")
    with pytest.raises(RuntimeError, match="requires either"):
        async for _ in agent.send_message(_make_message("hi")):
            pass


@pytest.mark.asyncio
async def test_remote_agent_get_card_from_client():
    mock_client = _mock_client()
    mock_client.get_card.return_value = {"name": "remote-agent"}
    agent = RemoteAgent(id="remote", client=mock_client)
    card = await agent.get_card()
    assert card == {"name": "remote-agent"}
