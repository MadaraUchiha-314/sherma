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


def _patch_shared_client(response_or_side_effect):
    """Patch ``get_http_client`` in ``sherma.hooks.remote`` to a mock client.

    Returns the mock client so tests can introspect calls.
    """
    mock_client = AsyncMock()
    if isinstance(response_or_side_effect, Exception) or (
        isinstance(response_or_side_effect, type)
        and issubclass(response_or_side_effect, Exception)
    ):
        mock_client.post = AsyncMock(side_effect=response_or_side_effect)
    else:
        mock_client.post = AsyncMock(return_value=response_or_side_effect)
    return mock_client, patch(
        "sherma.hooks.remote.get_http_client", AsyncMock(return_value=mock_client)
    )


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
    mock_client, ctx_patch = _patch_shared_client(mock_response)
    with ctx_patch:
        result = await executor.before_graph_invoke(ctx)

    assert result is not None
    assert result.config == {"k": "updated"}
    mock_client.post.assert_called_once()
    # Per-request timeout is applied from the executor's timeout setting
    assert mock_client.post.call_args.kwargs.get("timeout") == 5.0


@pytest.mark.asyncio
async def test_null_result_returns_none(executor):
    """JSON-RPC returning null result is treated as pass-through."""
    ctx = GraphInvokeContext(agent_id="a1", thread_id="t1", config={}, input={})
    mock_response = _make_jsonrpc_response(None)
    _, ctx_patch = _patch_shared_client(mock_response)
    with ctx_patch:
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
    _, ctx_patch = _patch_shared_client(httpx.ConnectError("connection refused"))

    with ctx_patch, patch("sherma.hooks.remote.logger") as mock_logger:
        result = await executor.before_graph_invoke(ctx)
        assert result is None
        mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_timeout_returns_none(executor):
    """Timeout logs warning and returns None."""
    ctx = GraphInvokeContext(agent_id="a1", thread_id="t1", config={}, input={})
    _, ctx_patch = _patch_shared_client(httpx.TimeoutException("timed out"))

    with ctx_patch, patch("sherma.hooks.remote.logger") as mock_logger:
        result = await executor.before_graph_invoke(ctx)
        assert result is None
        mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_jsonrpc_error_returns_none(executor):
    """JSON-RPC error response logs warning and returns None."""
    ctx = GraphInvokeContext(agent_id="a1", thread_id="t1", config={}, input={})
    _, ctx_patch = _patch_shared_client(_make_jsonrpc_error())

    with ctx_patch, patch("sherma.hooks.remote.logger") as mock_logger:
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
    _, ctx_patch = _patch_shared_client(mock_response)
    with ctx_patch:
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
    _, ctx_patch = _patch_shared_client(mock_response)
    with ctx_patch:
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
    mock_client, ctx_patch = _patch_shared_client(mock_response)
    with ctx_patch:
        result = await executor.on_error(ctx)
    call_args = mock_client.post.call_args
    sent_payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert sent_payload["params"]["error"] == {
        "type": "RuntimeError",
        "message": "boom",
    }
    assert result is not None
    assert result.error is ctx.error


@pytest.mark.asyncio
async def test_request_id_increments(executor):
    """Each RPC call gets an incrementing request ID."""
    ctx = GraphInvokeContext(agent_id="a1", thread_id="t1", config={}, input={})
    mock_response = _make_jsonrpc_response(None)
    mock_client, ctx_patch = _patch_shared_client(mock_response)
    with ctx_patch:
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
async def test_uses_shared_http_client(executor):
    """Executor delegates to ``sherma.hooks.remote.get_http_client``."""
    ctx = GraphInvokeContext(agent_id="a1", thread_id="t1", config={}, input={})
    mock_client, ctx_patch = _patch_shared_client(_make_jsonrpc_response(None))
    with ctx_patch as mock_get:
        await executor.before_graph_invoke(ctx)
        mock_get.assert_awaited()
    mock_client.post.assert_called_once()
