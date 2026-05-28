"""Unit tests for ``MCPHookServer``."""

from __future__ import annotations

from typing import Any

import pytest

from sherma.hooks.handler import HookHandler
from sherma.hooks.mcp_server import HOOK_TOOL_PREFIX, MCPHookServer


class _BareHandler(HookHandler):
    """A handler that overrides nothing — all base methods (pass-through)."""


class _PartialHandler(HookHandler):
    async def before_llm_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return {**params, "marker": "before"}

    async def node_enter(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None


class _RaisingHandler(HookHandler):
    async def before_llm_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        raise RuntimeError("oops")


def test_bare_handler_registers_no_tools():
    server = MCPHookServer(handler=_BareHandler(), name="bare")
    assert server.registered_hooks == set()


def test_partial_handler_registers_only_overridden_hooks():
    server = MCPHookServer(handler=_PartialHandler(), name="partial")
    assert server.registered_hooks == {"before_llm_call", "node_enter"}


@pytest.mark.asyncio
async def test_registered_tool_names_use_prefix():
    server = MCPHookServer(handler=_PartialHandler(), name="partial")
    tools = await server.server.list_tools()
    names = {t.name for t in tools}
    assert f"{HOOK_TOOL_PREFIX}before_llm_call" in names
    assert f"{HOOK_TOOL_PREFIX}node_enter" in names


@pytest.mark.asyncio
async def test_tool_invocation_delegates_to_handler():
    server = MCPHookServer(handler=_PartialHandler())
    tool_name = f"{HOOK_TOOL_PREFIX}before_llm_call"
    result = await server.server.call_tool(
        tool_name, {"params": {"system_prompt": "hi"}}
    )
    # FastMCP returns (content_list, structured_output)
    if isinstance(result, tuple):
        _, structured = result
    else:
        structured = None
    assert structured is not None
    inner = structured.get("result", structured)
    assert inner["marker"] == "before"
    assert inner["system_prompt"] == "hi"


@pytest.mark.asyncio
async def test_tool_invocation_can_return_none():
    server = MCPHookServer(handler=_PartialHandler())
    tool_name = f"{HOOK_TOOL_PREFIX}node_enter"
    result = await server.server.call_tool(
        tool_name, {"params": {"node_name": "x", "node_type": "call_llm"}}
    )
    if isinstance(result, tuple):
        _, structured = result
    else:
        structured = None
    if structured is not None:
        assert structured.get("result") is None


@pytest.mark.asyncio
async def test_handler_exception_surfaces_as_tool_error():
    """When a handler raises, the MCP server reports an error."""
    from mcp.server.fastmcp.exceptions import ToolError

    server = MCPHookServer(handler=_RaisingHandler())
    tool_name = f"{HOOK_TOOL_PREFIX}before_llm_call"
    with pytest.raises(ToolError):
        await server.server.call_tool(tool_name, {"params": {}})


def test_overriding_all_supported_hooks_registers_all():
    """Override every dispatchable hook and verify they're all exposed."""

    class _AllHooks(HookHandler):
        async def before_llm_call(self, p):
            return p

        async def after_llm_call(self, p):
            return p

        async def before_tool_call(self, p):
            return p

        async def after_tool_call(self, p):
            return p

        async def before_agent_call(self, p):
            return p

        async def after_agent_call(self, p):
            return p

        async def before_skill_load(self, p):
            return p

        async def after_skill_load(self, p):
            return p

        async def before_skill_unload(self, p):
            return p

        async def after_skill_unload(self, p):
            return p

        async def node_enter(self, p):
            return p

        async def node_execute(self, p):
            return p

        async def node_exit(self, p):
            return p

        async def before_interrupt(self, p):
            return p

        async def after_interrupt(self, p):
            return p

        async def before_graph_invoke(self, p):
            return p

        async def after_graph_invoke(self, p):
            return p

        async def on_node_error(self, p):
            return p

        async def on_error(self, p):
            return p

    from sherma.hooks.apps.jsonrpc_handler import HOOK_METHODS

    server = MCPHookServer(handler=_AllHooks())
    assert server.registered_hooks == set(HOOK_METHODS)
