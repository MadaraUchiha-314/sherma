"""Tests for the JSON-RPC dispatch logic shared by hook server apps."""

from __future__ import annotations

from typing import Any

import pytest

from sherma.hooks.apps.jsonrpc_handler import HOOK_METHODS, dispatch_jsonrpc
from sherma.hooks.handler import HookHandler

# -- Consistency checks --


def test_hook_methods_matches_handler():
    """HOOK_METHODS contains exactly the methods defined on HookHandler."""
    handler_methods = {
        name
        for name in dir(HookHandler)
        if not name.startswith("_") and callable(getattr(HookHandler, name))
    }
    assert HOOK_METHODS == handler_methods


def test_hook_methods_count():
    """There are exactly 17 remote-capable hooks (all except on_chat_model_create)."""
    assert len(HOOK_METHODS) == 17
    assert "on_chat_model_create" not in HOOK_METHODS


# -- dispatch_jsonrpc tests --


class ModifyHooks(HookHandler):
    async def before_llm_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        params["system_prompt"] += " [modified]"
        return params

    async def node_enter(self, params: dict[str, Any]) -> None:
        return None


class ErrorHooks(HookHandler):
    async def before_llm_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        raise RuntimeError("handler failed")


@pytest.mark.asyncio
async def test_dispatch_known_hook_modifies():
    """Dispatch calls handler and returns modified result."""
    handler = ModifyHooks()
    body = {
        "jsonrpc": "2.0",
        "method": "before_llm_call",
        "params": {"system_prompt": "hello"},
        "id": 1,
    }
    resp = await dispatch_jsonrpc(handler, body)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["system_prompt"] == "hello [modified]"


@pytest.mark.asyncio
async def test_dispatch_known_hook_passthrough():
    """Hook returning None produces null result."""
    handler = ModifyHooks()
    body = {
        "jsonrpc": "2.0",
        "method": "node_enter",
        "params": {"node_name": "llm", "node_type": "call_llm"},
        "id": 2,
    }
    resp = await dispatch_jsonrpc(handler, body)
    assert resp["result"] is None
    assert resp["id"] == 2


@pytest.mark.asyncio
async def test_dispatch_unknown_method():
    """Unknown method name returns null result."""
    handler = ModifyHooks()
    body = {
        "jsonrpc": "2.0",
        "method": "not_a_hook",
        "params": {},
        "id": 3,
    }
    resp = await dispatch_jsonrpc(handler, body)
    assert resp["result"] is None
    assert resp["id"] == 3


@pytest.mark.asyncio
async def test_dispatch_empty_method():
    """Missing method field returns null result."""
    handler = ModifyHooks()
    body = {"jsonrpc": "2.0", "params": {}, "id": 4}
    resp = await dispatch_jsonrpc(handler, body)
    assert resp["result"] is None


@pytest.mark.asyncio
async def test_dispatch_handler_error():
    """Handler exception returns JSON-RPC error."""
    handler = ErrorHooks()
    body = {
        "jsonrpc": "2.0",
        "method": "before_llm_call",
        "params": {"system_prompt": "x"},
        "id": 5,
    }
    resp = await dispatch_jsonrpc(handler, body)
    assert "error" in resp
    assert resp["error"]["code"] == -32000
    assert "handler failed" in resp["error"]["message"]
    assert resp["id"] == 5


@pytest.mark.asyncio
async def test_dispatch_preserves_request_id():
    """Response id matches request id."""
    handler = ModifyHooks()
    for rid in [1, "abc", 999]:
        body = {
            "jsonrpc": "2.0",
            "method": "after_llm_call",
            "params": {},
            "id": rid,
        }
        resp = await dispatch_jsonrpc(handler, body)
        assert resp["id"] == rid


@pytest.mark.asyncio
async def test_dispatch_missing_params():
    """Missing params field defaults to empty dict."""
    handler = ModifyHooks()
    body = {"jsonrpc": "2.0", "method": "node_enter", "id": 6}
    resp = await dispatch_jsonrpc(handler, body)
    assert resp["result"] is None
    assert resp["id"] == 6


@pytest.mark.asyncio
async def test_dispatch_all_hook_methods():
    """Every method in HOOK_METHODS dispatches without error."""
    handler = HookHandler()
    for method in sorted(HOOK_METHODS):
        body = {
            "jsonrpc": "2.0",
            "method": method,
            "params": {},
            "id": 1,
        }
        resp = await dispatch_jsonrpc(handler, body)
        assert resp["result"] is None, f"{method} should pass through"
