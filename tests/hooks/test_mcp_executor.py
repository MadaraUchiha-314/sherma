"""Unit tests for ``MCPHookExecutor``."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sherma.hooks.mcp_executor import (
    HOOK_TOOL_PREFIX,
    MCPHookExecutor,
    MCPHookTransport,
)
from sherma.hooks.types import (
    AfterGraphInvokeContext,
    BeforeLLMCallContext,
    ChatModelCreateContext,
    NodeEnterContext,
    OnErrorContext,
)


def _stub_tool(name: str):
    return SimpleNamespace(name=name, inputSchema={}, description=None)


def _structured_result(result_dict):
    """Build a CallToolResult-like object using structuredContent."""
    return SimpleNamespace(
        isError=False,
        content=[],
        structuredContent={"result": result_dict},
    )


def _text_result(payload):
    """Build a CallToolResult-like object using text content."""
    text = json.dumps(payload)
    return SimpleNamespace(
        isError=False,
        content=[SimpleNamespace(text=text)],
        structuredContent=None,
    )


def _error_result(message="bad"):
    return SimpleNamespace(
        isError=True,
        content=[SimpleNamespace(text=message)],
        structuredContent=None,
    )


def _make_executor(available: set[str] | None = None) -> MCPHookExecutor:
    """Build an MCPHookExecutor with a mocked ``ClientSession``."""
    transport = MCPHookTransport(url="http://localhost:9999/mcp")
    executor = MCPHookExecutor(transport)
    executor._session = AsyncMock()
    executor._available_hooks = (
        set(available) if available is not None else {"before_llm_call"}
    )
    return executor


# ---------------------------------------------------------------------------
# Transport validation
# ---------------------------------------------------------------------------


def test_transport_requires_one_source():
    with pytest.raises(ValueError):
        MCPHookTransport()
    with pytest.raises(ValueError):
        MCPHookTransport(command="x", url="http://y")


def test_transport_stdio_ok():
    t = MCPHookTransport(command="python", args=["-m", "srv"])
    assert t.command == "python"
    assert t.url is None


def test_transport_http_ok():
    t = MCPHookTransport(url="http://localhost:9000/mcp")
    assert t.url == "http://localhost:9000/mcp"
    assert t.transport == "streamable_http"


# ---------------------------------------------------------------------------
# Hook dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_when_hook_not_available():
    """Hook not exposed by the server → no tool call, return None."""
    executor = _make_executor(available={"node_enter"})
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="x",
        messages=[],
        system_prompt="hi",
        tools=[],
        state={},
    )
    result = await executor.before_llm_call(ctx)
    assert result is None
    executor._session.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_passes_through_on_null_structured_result():
    executor = _make_executor(available={"before_llm_call"})
    executor._session.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            isError=False, content=[], structuredContent={"result": None}
        )
    )
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="x",
        messages=[],
        system_prompt="hi",
        tools=[],
        state={},
    )
    result = await executor.before_llm_call(ctx)
    assert result is None


@pytest.mark.asyncio
async def test_modified_context_via_structured_result():
    executor = _make_executor(available={"before_llm_call"})
    executor._session.call_tool = AsyncMock(
        return_value=_structured_result(
            {"node_name": "x", "system_prompt": "modified", "state": {"k": 1}}
        )
    )
    nc = MagicMock()
    messages = [MagicMock()]
    tools = [MagicMock()]
    ctx = BeforeLLMCallContext(
        node_context=nc,
        node_name="x",
        messages=messages,
        system_prompt="orig",
        tools=tools,
        state={},
    )
    result = await executor.before_llm_call(ctx)
    assert result is not None
    assert result.system_prompt == "modified"
    assert result.state == {"k": 1}
    # Non-serializable / complex fields are preserved from original
    assert result.node_context is nc
    assert result.messages is messages
    assert result.tools is tools


@pytest.mark.asyncio
async def test_text_content_fallback():
    """If the server returns text content (no structuredContent), it is parsed."""
    executor = _make_executor(available={"before_llm_call"})
    executor._session.call_tool = AsyncMock(
        return_value=_text_result(
            {"node_name": "x", "system_prompt": "via-text", "state": {}}
        )
    )
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="x",
        messages=[],
        system_prompt="orig",
        tools=[],
        state={},
    )
    result = await executor.before_llm_call(ctx)
    assert result is not None
    assert result.system_prompt == "via-text"


@pytest.mark.asyncio
async def test_tool_call_uses_prefixed_name():
    executor = _make_executor(available={"before_llm_call"})
    executor._session.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            isError=False, content=[], structuredContent={"result": None}
        )
    )
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="x",
        messages=[],
        system_prompt="hi",
        tools=[],
        state={},
    )
    await executor.before_llm_call(ctx)
    call = executor._session.call_tool.await_args
    assert call.args[0] == f"{HOOK_TOOL_PREFIX}before_llm_call"
    sent_params = call.args[1]["params"]
    assert sent_params["system_prompt"] == "hi"
    assert sent_params["node_name"] == "x"
    assert "node_context" not in sent_params  # non-serializable, stripped


@pytest.mark.asyncio
async def test_call_tool_exception_returns_none():
    executor = _make_executor(available={"before_llm_call"})
    executor._session.call_tool = AsyncMock(side_effect=RuntimeError("oops"))

    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="x",
        messages=[],
        system_prompt="hi",
        tools=[],
        state={},
    )
    with patch("sherma.hooks.mcp_executor.logger") as mock_logger:
        result = await executor.before_llm_call(ctx)
        assert result is None
        mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_error_result_returns_none():
    executor = _make_executor(available={"before_llm_call"})
    executor._session.call_tool = AsyncMock(return_value=_error_result("failed"))
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="x",
        messages=[],
        system_prompt="hi",
        tools=[],
        state={},
    )
    with patch("sherma.hooks.mcp_executor.logger") as mock_logger:
        result = await executor.before_llm_call(ctx)
        assert result is None
        mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_invalid_json_text_returns_none():
    executor = _make_executor(available={"before_llm_call"})
    executor._session.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(text="not json")],
            structuredContent=None,
        )
    )
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="x",
        messages=[],
        system_prompt="hi",
        tools=[],
        state={},
    )
    with patch("sherma.hooks.mcp_executor.logger") as mock_logger:
        result = await executor.before_llm_call(ctx)
        assert result is None
        mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_session_not_connected_returns_none():
    executor = MCPHookExecutor(MCPHookTransport(url="http://localhost:1/mcp"))
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="x",
        messages=[],
        system_prompt="hi",
        tools=[],
        state={},
    )
    # Available hooks empty → still skipped without warning about session.
    result = await executor.before_llm_call(ctx)
    assert result is None


@pytest.mark.asyncio
async def test_on_chat_model_create_is_noop():
    executor = _make_executor(available={"on_chat_model_create"})
    ctx = ChatModelCreateContext(
        llm_id="x", provider="openai", model_name="gpt-4", kwargs={}
    )
    result = await executor.on_chat_model_create(ctx)
    assert result is None


@pytest.mark.asyncio
async def test_node_enter_strips_node_context():
    executor = _make_executor(available={"node_enter"})
    executor._session.call_tool = AsyncMock(
        return_value=_structured_result(
            {"node_name": "x", "node_type": "call_llm", "state": {"v": 2}}
        )
    )
    nc = MagicMock()
    ctx = NodeEnterContext(
        node_context=nc, node_name="x", node_type="call_llm", state={"v": 1}
    )
    result = await executor.node_enter(ctx)
    assert result is not None
    assert result.node_context is nc
    assert result.state == {"v": 2}


@pytest.mark.asyncio
async def test_on_error_serializes_error_in_params():
    executor = _make_executor(available={"on_error"})
    executor._session.call_tool = AsyncMock(
        return_value=_structured_result(
            {
                "agent_id": "a",
                "thread_id": "t",
                "config": {},
                "input": {},
            }
        )
    )
    err = RuntimeError("boom")
    ctx = OnErrorContext(agent_id="a", thread_id="t", config={}, input={}, error=err)
    result = await executor.on_error(ctx)
    call = executor._session.call_tool.await_args
    sent_params = call.args[1]["params"]
    assert sent_params["error"] == {"type": "RuntimeError", "message": "boom"}
    # The error object itself is re-attached from the original
    assert result is not None
    assert result.error is err


@pytest.mark.asyncio
async def test_after_graph_invoke_dispatch():
    executor = _make_executor(available={"after_graph_invoke"})
    executor._session.call_tool = AsyncMock(
        return_value=_structured_result(
            {
                "agent_id": "a",
                "thread_id": "t",
                "config": {"recursion_limit": 99},
                "input": {},
                "result": {"ok": True},
            }
        )
    )
    ctx = AfterGraphInvokeContext(
        agent_id="a", thread_id="t", config={}, input={}, result={}
    )
    result = await executor.after_graph_invoke(ctx)
    assert result is not None
    assert result.config["recursion_limit"] == 99


@pytest.mark.asyncio
async def test_available_hooks_property():
    executor = _make_executor(available={"node_enter", "node_exit"})
    assert executor.available_hooks == {"node_enter", "node_exit"}
    # Returned set is a copy
    executor.available_hooks.add("xx")
    assert "xx" not in executor.available_hooks


@pytest.mark.asyncio
async def test_close_is_idempotent():
    executor = MCPHookExecutor(MCPHookTransport(url="http://localhost:1/mcp"))
    # Without connect, close is a no-op
    await executor.close()
    await executor.close()


@pytest.mark.asyncio
async def test_close_clears_session_and_hooks():
    executor = _make_executor(available={"node_enter"})
    mock_stack = AsyncMock()
    executor._exit_stack = mock_stack
    await executor.close()
    mock_stack.aclose.assert_awaited_once()
    assert executor._session is None
    assert executor._exit_stack is None
    assert executor._available_hooks == set()


@pytest.mark.asyncio
async def test_tool_discovery_filters_non_hook_tools():
    """Only tools with the hooks. prefix are considered hooks."""
    executor = MCPHookExecutor(MCPHookTransport(url="http://localhost:1/mcp"))

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(
        return_value=SimpleNamespace(
            tools=[
                _stub_tool("hooks.before_llm_call"),
                _stub_tool("hooks.after_llm_call"),
                _stub_tool("unrelated_tool"),
            ]
        )
    )

    # Patch the AsyncExitStack to return the transport tuple, then the session.
    with patch(
        "contextlib.AsyncExitStack.enter_async_context",
        side_effect=[
            (AsyncMock(), AsyncMock(), AsyncMock()),  # streamablehttp_client returns 3
            mock_session,
        ],
    ):
        await executor.connect()

    assert executor.available_hooks == {"before_llm_call", "after_llm_call"}
    await executor.close()
