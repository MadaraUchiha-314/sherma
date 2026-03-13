"""Integration test: RemoteHookExecutor <-> HookStarletteApplication over HTTP.

Starts a real Starlette hook server on a random port, then uses
RemoteHookExecutor to send hooks through the full HTTP stack.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
import uvicorn

from sherma.hooks.apps import HookStarletteApplication
from sherma.hooks.handler import HookHandler
from sherma.hooks.remote import RemoteHookExecutor
from sherma.hooks.types import (
    BeforeLLMCallContext,
    ChatModelCreateContext,
    GraphInvokeContext,
    NodeEnterContext,
    OnErrorContext,
)

# -- Hook handler for the server side --


class TestHooks(HookHandler):
    """Server-side handler used during integration tests."""

    async def before_llm_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        params["system_prompt"] += " [from-server]"
        return params

    async def before_graph_invoke(
        self, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        params["config"]["recursion_limit"] = 77
        return params

    async def node_enter(self, params: dict[str, Any]) -> None:
        # Observation only
        return None

    async def on_error(self, params: dict[str, Any]) -> dict[str, Any] | None:
        params["agent_id"] = params["agent_id"] + "-handled"
        return params


# -- Fixtures --


@pytest.fixture(scope="module")
def hook_server_url():
    """Start a Starlette hook server in a background thread, return its URL."""
    app = HookStarletteApplication(handler=TestHooks()).build()

    # Use port 0 to let OS pick a free port; uvicorn doesn't expose it
    # directly, so we use a known free port via socket.
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to start
    import time

    for _ in range(50):
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=0.1)
            sock.close()
            break
        except OSError:
            time.sleep(0.1)

    url = f"http://127.0.0.1:{port}/hooks"
    yield url

    server.should_exit = True
    thread.join(timeout=3)


@pytest.fixture
def executor(hook_server_url):
    return RemoteHookExecutor(url=hook_server_url, timeout=5.0)


# -- Tests --


@pytest.mark.integration
@pytest.mark.asyncio
async def test_before_llm_call_modifies_prompt(executor):
    """Full round-trip: executor serializes, server modifies, executor deserializes."""
    nc = MagicMock()
    ctx = BeforeLLMCallContext(
        node_context=nc,
        node_name="agent",
        messages=[],
        system_prompt="Be helpful",
        tools=[],
        state={"messages": []},
    )
    result = await executor.before_llm_call(ctx)

    assert result is not None
    assert result.system_prompt == "Be helpful [from-server]"
    # node_context re-attached from original
    assert result.node_context is nc
    # messages/tools kept from original
    assert result.messages is ctx.messages
    assert result.tools is ctx.tools


@pytest.mark.integration
@pytest.mark.asyncio
async def test_before_graph_invoke_modifies_config(executor):
    """Server modifies config dict and executor picks it up."""
    ctx = GraphInvokeContext(
        agent_id="test-agent",
        thread_id="t1",
        config={"recursion_limit": 25},
        input={"messages": []},
    )
    result = await executor.before_graph_invoke(ctx)

    assert result is not None
    assert result.config["recursion_limit"] == 77


@pytest.mark.integration
@pytest.mark.asyncio
async def test_node_enter_passthrough(executor):
    """Server returns None for observation-only hook -> executor returns None."""
    nc = MagicMock()
    ctx = NodeEnterContext(
        node_context=nc,
        node_name="llm1",
        node_type="call_llm",
        state={},
    )
    result = await executor.node_enter(ctx)
    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_on_chat_model_create_noop(executor):
    """on_chat_model_create never hits the server — always None."""
    ctx = ChatModelCreateContext(
        llm_id="llm1",
        provider="openai",
        model_name="gpt-4",
        kwargs={},
    )
    result = await executor.on_chat_model_create(ctx)
    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unhandled_hook_passthrough(executor):
    """Hook not implemented on server passes through."""
    nc = MagicMock()
    # after_llm_call is NOT overridden in TestHooks, so it should pass through
    from sherma.hooks.types import AfterLLMCallContext

    after_ctx = AfterLLMCallContext(
        node_context=nc,
        node_name="agent",
        response=MagicMock(),
        state={"k": 1},
    )
    result = await executor.after_llm_call(after_ctx)
    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_on_error_modifies_agent_id(executor):
    """Error hook round-trip: error field serialized, agent_id modified."""
    ctx = OnErrorContext(
        agent_id="my-agent",
        thread_id="t1",
        config={},
        input={},
        error=RuntimeError("boom"),
    )
    result = await executor.on_error(ctx)

    assert result is not None
    assert result.agent_id == "my-agent-handled"
    # error is re-attached from original, not deserialized from server
    assert result.error is ctx.error


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_hooks_sequentially(executor):
    """Multiple different hooks work in sequence on the same executor."""
    nc = MagicMock()

    # 1. before_llm_call
    ctx1 = BeforeLLMCallContext(
        node_context=nc,
        node_name="n",
        messages=[],
        system_prompt="A",
        tools=[],
        state={},
    )
    r1 = await executor.before_llm_call(ctx1)
    assert r1 is not None
    assert r1.system_prompt == "A [from-server]"

    # 2. before_graph_invoke
    ctx2 = GraphInvokeContext(
        agent_id="a", thread_id="t", config={"recursion_limit": 10}, input={}
    )
    r2 = await executor.before_graph_invoke(ctx2)
    assert r2 is not None
    assert r2.config["recursion_limit"] == 77

    # 3. node_enter (passthrough)
    ctx3 = NodeEnterContext(
        node_context=nc, node_name="x", node_type="tool_node", state={}
    )
    r3 = await executor.node_enter(ctx3)
    assert r3 is None
