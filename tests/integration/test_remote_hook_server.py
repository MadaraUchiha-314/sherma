"""Integration test: remote JSON-RPC hook server dispatch.

Tests the JSON-RPC dispatch logic directly (no FastAPI dependency needed).
Mirrors examples/remote_hook_server/server.py pattern — server-side hooks
are authored as a typed ``BaseHookExecutor``.
"""

from __future__ import annotations

from typing import Any

import pytest

from sherma.hooks.apps.jsonrpc_handler import dispatch_jsonrpc
from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.types import (
    BeforeLLMCallContext,
    GraphInvokeContext,
    NodeEnterContext,
)


class TestHookExecutor(BaseHookExecutor):
    """Hook executor for testing — same pattern as the example."""

    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        ctx.system_prompt += "\n\nALWAYS be helpful and concise."
        return ctx

    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
        # Logging hook: returns None (pass-through)
        return None

    async def before_graph_invoke(
        self, ctx: GraphInvokeContext
    ) -> GraphInvokeContext | None:
        ctx.config["recursion_limit"] = 50
        return ctx


@pytest.fixture
def executor():
    return TestHookExecutor()


def _jsonrpc_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_before_llm_call_modifies_prompt(executor):
    """before_llm_call hook appends text to system_prompt."""
    body = _jsonrpc_request(
        "before_llm_call",
        {
            "system_prompt": "You are a weather assistant.",
            "messages": [],
            "tools": [],
            "node_name": "agent",
            "state": {},
        },
    )
    resp = await dispatch_jsonrpc(executor, body)
    assert resp["result"] is not None
    assert "ALWAYS be helpful" in resp["result"]["system_prompt"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_before_graph_invoke_sets_recursion_limit(executor):
    """before_graph_invoke sets recursion_limit to 50."""
    body = _jsonrpc_request(
        "before_graph_invoke",
        {
            "agent_id": "test-agent",
            "thread_id": "t1",
            "config": {"recursion_limit": 25},
            "input": {},
        },
    )
    resp = await dispatch_jsonrpc(executor, body)
    assert resp["result"] is not None
    assert resp["result"]["config"]["recursion_limit"] == 50


@pytest.mark.integration
@pytest.mark.asyncio
async def test_node_enter_returns_null(executor):
    """node_enter (logging) returns None → JSON-RPC result is null."""
    body = _jsonrpc_request(
        "node_enter",
        {
            "node_name": "agent",
            "node_type": "call_llm",
            "state": {},
        },
    )
    resp = await dispatch_jsonrpc(executor, body)
    assert resp["result"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unknown_method_returns_null(executor):
    """An unknown hook method returns null result (not an error)."""
    body = _jsonrpc_request("unknown_method", {})
    resp = await dispatch_jsonrpc(executor, body)
    assert resp["result"] is None
