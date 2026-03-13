"""Tests for RemoteHookExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from sherma.hooks.remote import RemoteHookExecutor
from sherma.hooks.types import (
    AfterGraphInvokeContext,
    BeforeLLMCallContext,
    ChatModelCreateContext,
    GraphInvokeContext,
    NodeEnterContext,
    OnErrorContext,
)


@pytest.fixture
def executor():
    return RemoteHookExecutor(url="http://localhost:9999/hooks", timeout=5.0)


_FAKE_REQUEST = httpx.Request("POST", "http://localhost:9999/hooks")


def _make_jsonrpc_response(result, request_id=1):
    """Build an httpx.Response wrapping a JSON-RPC success result."""
    body = {"jsonrpc": "2.0", "result": result, "id": request_id}
    return httpx.Response(200, json=body, request=_FAKE_REQUEST)


def _make_jsonrpc_error(code=-32000, message="server error", request_id=1):
    body = {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": request_id,
    }
    return httpx.Response(200, json=body, request=_FAKE_REQUEST)


@pytest.mark.asyncio
async def test_successful_rpc_call(executor):
    """Successful JSON-RPC call updates the context."""
    ctx = GraphInvokeContext(
        agent_id="a1", thread_id="t1", config={"k": "v"}, input={"msg": "hi"}
    )
    mock_response = _make_jsonrpc_response(
        {
            "agent_id": "a1",
            "thread_id": "t1",
            "config": {"k": "updated"},
            "input": {"msg": "hi"},
        }
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    executor._client = mock_client

    result = await executor.before_graph_invoke(ctx)
    assert result is not None
    assert result.config == {"k": "updated"}
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_null_result_returns_none(executor):
    """JSON-RPC returning null result is treated as pass-through."""
    ctx = GraphInvokeContext(agent_id="a1", thread_id="t1", config={}, input={})
    mock_response = _make_jsonrpc_response(None)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    executor._client = mock_client

    result = await executor.before_graph_invoke(ctx)
    assert result is None


@pytest.mark.asyncio
async def test_on_chat_model_create_is_noop(executor):
    """on_chat_model_create always returns None for remote hooks."""
    ctx = ChatModelCreateContext(
        llm_id="llm1", provider="openai", model_name="gpt-4", kwargs={}
    )
    result = await executor.on_chat_model_create(ctx)
    assert result is None


@pytest.mark.asyncio
async def test_network_error_returns_none(executor):
    """Network error logs warning and returns None."""
    ctx = GraphInvokeContext(agent_id="a1", thread_id="t1", config={}, input={})
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    executor._client = mock_client

    with patch("sherma.hooks.remote.logger") as mock_logger:
        result = await executor.before_graph_invoke(ctx)
        assert result is None
        mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_timeout_returns_none(executor):
    """Timeout logs warning and returns None."""
    ctx = GraphInvokeContext(agent_id="a1", thread_id="t1", config={}, input={})
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
    executor._client = mock_client

    with patch("sherma.hooks.remote.logger") as mock_logger:
        result = await executor.before_graph_invoke(ctx)
        assert result is None
        mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_jsonrpc_error_returns_none(executor):
    """JSON-RPC error response logs warning and returns None."""
    ctx = GraphInvokeContext(agent_id="a1", thread_id="t1", config={}, input={})
    mock_response = _make_jsonrpc_error()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    executor._client = mock_client

    with patch("sherma.hooks.remote.logger") as mock_logger:
        result = await executor.before_graph_invoke(ctx)
        assert result is None
        mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_node_enter_with_node_context(executor):
    """Node-level hooks strip and re-attach node_context."""
    nc = MagicMock()
    ctx = NodeEnterContext(
        node_context=nc, node_name="llm1", node_type="call_llm", state={"k": 1}
    )
    mock_response = _make_jsonrpc_response(
        {"node_name": "llm1", "node_type": "call_llm", "state": {"k": 2}}
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    executor._client = mock_client

    result = await executor.node_enter(ctx)
    assert result is not None
    assert result.node_context is nc
    assert result.state == {"k": 2}


@pytest.mark.asyncio
async def test_before_llm_call_preserves_complex_fields(executor):
    """Complex fields like messages/tools are kept from original."""
    nc = MagicMock()
    messages = [MagicMock()]
    tools = [MagicMock()]
    ctx = BeforeLLMCallContext(
        node_context=nc,
        node_name="llm1",
        messages=messages,
        system_prompt="old",
        tools=tools,
        state={"k": 1},
    )
    mock_response = _make_jsonrpc_response(
        {"node_name": "llm1", "system_prompt": "new", "state": {"k": 2}}
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    executor._client = mock_client

    result = await executor.before_llm_call(ctx)
    assert result is not None
    assert result.node_context is nc
    assert result.messages is messages
    assert result.tools is tools
    assert result.system_prompt == "new"
    assert result.state == {"k": 2}


@pytest.mark.asyncio
async def test_on_error_serializes_error(executor):
    """Error hooks serialize the error field."""
    ctx = OnErrorContext(
        agent_id="a1",
        thread_id="t1",
        config={},
        input={},
        error=RuntimeError("boom"),
    )
    mock_response = _make_jsonrpc_response(
        {"agent_id": "a1", "thread_id": "t1", "config": {}, "input": {}}
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    executor._client = mock_client

    # Verify the error was serialized in the RPC call
    result = await executor.on_error(ctx)
    call_args = mock_client.post.call_args
    sent_payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert sent_payload["params"]["error"] == {
        "type": "RuntimeError",
        "message": "boom",
    }
    # Error is re-attached from original
    assert result is not None
    assert result.error is ctx.error


@pytest.mark.asyncio
async def test_request_id_increments(executor):
    """Each RPC call gets an incrementing request ID."""
    ctx = GraphInvokeContext(agent_id="a1", thread_id="t1", config={}, input={})
    mock_response = _make_jsonrpc_response(None)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    executor._client = mock_client

    await executor.before_graph_invoke(ctx)
    await executor.after_graph_invoke(
        AfterGraphInvokeContext(
            agent_id="a1", thread_id="t1", config={}, input={}, result={}
        )
    )

    calls = mock_client.post.call_args_list
    id1 = calls[0].kwargs.get("json", calls[0][1].get("json", {})).get("id")
    id2 = calls[1].kwargs.get("json", calls[1][1].get("json", {})).get("id")
    assert id1 == 1
    assert id2 == 2


@pytest.mark.asyncio
async def test_lazy_client_creation(executor):
    """httpx client is created lazily on first call."""
    assert executor._client is None
    ctx = GraphInvokeContext(agent_id="a1", thread_id="t1", config={}, input={})
    # Patch httpx.AsyncClient to avoid real network calls
    with patch("sherma.hooks.remote.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=_make_jsonrpc_response(None))
        mock_cls.return_value = mock_instance

        await executor.before_graph_invoke(ctx)
        mock_cls.assert_called_once_with(timeout=5.0)
