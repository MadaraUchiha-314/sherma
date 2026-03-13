"""Tests for HookFastAPIApplication and HookStarletteApplication."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from starlette.testclient import TestClient

from sherma.hooks.apps import HookFastAPIApplication, HookStarletteApplication
from sherma.hooks.handler import HookHandler

_fastapi_installed = importlib.util.find_spec("fastapi") is not None


class EchoHooks(HookHandler):
    """Handler that modifies system_prompt on before_llm_call."""

    async def before_llm_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        params["system_prompt"] += " [modified]"
        return params

    async def before_graph_invoke(
        self, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        params["config"]["recursion_limit"] = 99
        return params


class ErrorHooks(HookHandler):
    """Handler that raises on before_llm_call."""

    async def before_llm_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        raise ValueError("hook exploded")


def _rpc_request(method: str, params: dict[str, Any], id: int = 1):
    return {"jsonrpc": "2.0", "method": method, "params": params, "id": id}


# --- FastAPI tests ---


@pytest.mark.skipif(not _fastapi_installed, reason="fastapi not installed")
class TestHookFastAPIApplication:
    def test_build_returns_app(self):
        app = HookFastAPIApplication(handler=EchoHooks()).build()
        assert app is not None

    def test_hook_modifies_context(self):
        app = HookFastAPIApplication(handler=EchoHooks()).build()
        client = TestClient(app)
        resp = client.post(
            "/hooks",
            json=_rpc_request(
                "before_llm_call",
                {"system_prompt": "hello", "state": {}},
            ),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["system_prompt"] == "hello [modified]"
        assert body["id"] == 1

    def test_unhandled_hook_passes_through(self):
        app = HookFastAPIApplication(handler=EchoHooks()).build()
        client = TestClient(app)
        resp = client.post(
            "/hooks",
            json=_rpc_request("on_error", {"agent_id": "a1"}),
        )
        body = resp.json()
        assert body["result"] is None

    def test_unknown_method_passes_through(self):
        app = HookFastAPIApplication(handler=EchoHooks()).build()
        client = TestClient(app)
        resp = client.post(
            "/hooks",
            json=_rpc_request("nonexistent_hook", {}),
        )
        body = resp.json()
        assert body["result"] is None

    def test_handler_error_returns_jsonrpc_error(self):
        app = HookFastAPIApplication(handler=ErrorHooks()).build()
        client = TestClient(app)
        resp = client.post(
            "/hooks",
            json=_rpc_request("before_llm_call", {"system_prompt": "x"}),
        )
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == -32000
        assert "hook exploded" in body["error"]["message"]

    def test_custom_rpc_url(self):
        app = HookFastAPIApplication(handler=EchoHooks()).build(rpc_url="/custom")
        client = TestClient(app)
        resp = client.post(
            "/custom",
            json=_rpc_request("before_llm_call", {"system_prompt": "hi"}),
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["system_prompt"] == "hi [modified]"

    def test_multiple_hooks(self):
        app = HookFastAPIApplication(handler=EchoHooks()).build()
        client = TestClient(app)

        resp1 = client.post(
            "/hooks",
            json=_rpc_request("before_llm_call", {"system_prompt": "a"}, id=1),
        )
        resp2 = client.post(
            "/hooks",
            json=_rpc_request(
                "before_graph_invoke",
                {"config": {"recursion_limit": 25}},
                id=2,
            ),
        )

        assert resp1.json()["result"]["system_prompt"] == "a [modified]"
        r2 = resp2.json()["result"]
        assert r2["config"]["recursion_limit"] == 99


# --- Starlette tests ---


class TestHookStarletteApplication:
    def test_build_returns_app(self):
        app = HookStarletteApplication(handler=EchoHooks()).build()
        assert app is not None

    def test_hook_modifies_context(self):
        app = HookStarletteApplication(handler=EchoHooks()).build()
        client = TestClient(app)
        resp = client.post(
            "/hooks",
            json=_rpc_request(
                "before_llm_call",
                {"system_prompt": "hello", "state": {}},
            ),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["system_prompt"] == "hello [modified]"
        assert body["id"] == 1

    def test_unhandled_hook_passes_through(self):
        app = HookStarletteApplication(handler=EchoHooks()).build()
        client = TestClient(app)
        resp = client.post(
            "/hooks",
            json=_rpc_request("on_error", {"agent_id": "a1"}),
        )
        body = resp.json()
        assert body["result"] is None

    def test_unknown_method_passes_through(self):
        app = HookStarletteApplication(handler=EchoHooks()).build()
        client = TestClient(app)
        resp = client.post(
            "/hooks",
            json=_rpc_request("nonexistent_hook", {}),
        )
        body = resp.json()
        assert body["result"] is None

    def test_handler_error_returns_jsonrpc_error(self):
        app = HookStarletteApplication(handler=ErrorHooks()).build()
        client = TestClient(app)
        resp = client.post(
            "/hooks",
            json=_rpc_request("before_llm_call", {"system_prompt": "x"}),
        )
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == -32000
        assert "hook exploded" in body["error"]["message"]

    def test_custom_rpc_url(self):
        app = HookStarletteApplication(handler=EchoHooks()).build(rpc_url="/custom")
        client = TestClient(app)
        resp = client.post(
            "/custom",
            json=_rpc_request("before_llm_call", {"system_prompt": "hi"}),
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["system_prompt"] == "hi [modified]"

    def test_multiple_hooks(self):
        app = HookStarletteApplication(handler=EchoHooks()).build()
        client = TestClient(app)

        resp1 = client.post(
            "/hooks",
            json=_rpc_request("before_llm_call", {"system_prompt": "a"}, id=1),
        )
        resp2 = client.post(
            "/hooks",
            json=_rpc_request(
                "before_graph_invoke",
                {"config": {"recursion_limit": 25}},
                id=2,
            ),
        )

        assert resp1.json()["result"]["system_prompt"] == "a [modified]"
        r2 = resp2.json()["result"]
        assert r2["config"]["recursion_limit"] == 99

    def test_routes_returns_list(self):
        hook_app = HookStarletteApplication(handler=EchoHooks())
        routes = hook_app.routes()
        assert len(routes) == 1
        assert routes[0].path == "/hooks"

    def test_routes_custom_url(self):
        hook_app = HookStarletteApplication(handler=EchoHooks())
        routes = hook_app.routes(rpc_url="/my-rpc")
        assert routes[0].path == "/my-rpc"

    def test_add_routes_to_existing_app(self):
        from starlette.applications import Starlette

        existing_app = Starlette()
        hook_app = HookStarletteApplication(handler=EchoHooks())
        hook_app.add_routes_to_app(existing_app, rpc_url="/my-hooks")

        client = TestClient(existing_app)
        resp = client.post(
            "/my-hooks",
            json=_rpc_request("before_llm_call", {"system_prompt": "test"}),
        )
        assert resp.json()["result"]["system_prompt"] == "test [modified]"

    def test_build_passes_kwargs_to_starlette(self):
        app = HookStarletteApplication(handler=EchoHooks()).build(debug=True)
        assert app.debug is True
