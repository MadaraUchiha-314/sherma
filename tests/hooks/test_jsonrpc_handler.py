"""Tests for the server-side hook dispatch shared by hook server apps."""

from __future__ import annotations

import pytest

from sherma.hooks.apps.jsonrpc_handler import HOOK_METHODS, dispatch_jsonrpc
from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.types import (
    BeforeLLMCallContext,
    NodeEnterContext,
)

# -- Consistency checks --


def test_hook_methods_matches_base_executor():
    """HOOK_METHODS is the BaseHookExecutor hooks minus on_chat_model_create."""
    executor_methods = {
        name
        for name in dir(BaseHookExecutor)
        if not name.startswith("_") and callable(getattr(BaseHookExecutor, name))
    }
    assert HOOK_METHODS == executor_methods - {"on_chat_model_create"}


def test_hook_methods_count():
    """There are exactly 19 remote-capable hooks (all except on_chat_model_create)."""
    assert len(HOOK_METHODS) == 19
    assert "on_chat_model_create" not in HOOK_METHODS


# -- dispatch_jsonrpc tests --


class ModifyHooks(BaseHookExecutor):
    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        ctx.system_prompt += " [modified]"
        return ctx

    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
        return None


class ErrorHooks(BaseHookExecutor):
    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        raise RuntimeError("handler failed")


@pytest.mark.asyncio
async def test_dispatch_known_hook_modifies():
    """Dispatch builds the typed context, runs the hook, serializes the result."""
    executor = ModifyHooks()
    body = {
        "jsonrpc": "2.0",
        "method": "before_llm_call",
        "params": {"system_prompt": "hello", "node_name": "agent", "state": {}},
        "id": 1,
    }
    resp = await dispatch_jsonrpc(executor, body)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["system_prompt"] == "hello [modified]"


@pytest.mark.asyncio
async def test_dispatch_known_hook_passthrough():
    """Hook returning None produces null result."""
    executor = ModifyHooks()
    body = {
        "jsonrpc": "2.0",
        "method": "node_enter",
        "params": {"node_name": "llm", "node_type": "call_llm", "state": {}},
        "id": 2,
    }
    resp = await dispatch_jsonrpc(executor, body)
    assert resp["result"] is None
    assert resp["id"] == 2


@pytest.mark.asyncio
async def test_dispatch_unknown_method():
    """Unknown method name returns null result."""
    executor = ModifyHooks()
    body = {"jsonrpc": "2.0", "method": "not_a_hook", "params": {}, "id": 3}
    resp = await dispatch_jsonrpc(executor, body)
    assert resp["result"] is None
    assert resp["id"] == 3


@pytest.mark.asyncio
async def test_dispatch_empty_method():
    """Missing method field returns null result."""
    executor = ModifyHooks()
    body = {"jsonrpc": "2.0", "params": {}, "id": 4}
    resp = await dispatch_jsonrpc(executor, body)
    assert resp["result"] is None


@pytest.mark.asyncio
async def test_dispatch_handler_error():
    """Hook exception returns JSON-RPC error."""
    executor = ErrorHooks()
    body = {
        "jsonrpc": "2.0",
        "method": "before_llm_call",
        "params": {"system_prompt": "x", "node_name": "n", "state": {}},
        "id": 5,
    }
    resp = await dispatch_jsonrpc(executor, body)
    assert "error" in resp
    assert resp["error"]["code"] == -32000
    assert "handler failed" in resp["error"]["message"]
    assert resp["id"] == 5


@pytest.mark.asyncio
async def test_dispatch_preserves_request_id():
    """Response id matches request id."""
    executor = ModifyHooks()
    for rid in [1, "abc", 999]:
        body = {
            "jsonrpc": "2.0",
            "method": "after_llm_call",
            "params": {"node_name": "n", "state": {}},
            "id": rid,
        }
        resp = await dispatch_jsonrpc(executor, body)
        assert resp["id"] == rid


@pytest.mark.asyncio
async def test_dispatch_missing_params():
    """Missing params field defaults to empty dict."""
    executor = ModifyHooks()
    body = {"jsonrpc": "2.0", "method": "node_enter", "id": 6}
    resp = await dispatch_jsonrpc(executor, body)
    assert resp["result"] is None
    assert resp["id"] == 6


@pytest.mark.asyncio
async def test_dispatch_all_hook_methods():
    """Every method in HOOK_METHODS dispatches without error (base passes through)."""
    executor = BaseHookExecutor()
    for method in sorted(HOOK_METHODS):
        body = {"jsonrpc": "2.0", "method": method, "params": {}, "id": 1}
        resp = await dispatch_jsonrpc(executor, body)
        assert resp["result"] is None, f"{method} should pass through"
