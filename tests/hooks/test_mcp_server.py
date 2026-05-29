"""Unit tests for ``MCPHookServer``."""

from __future__ import annotations

import pytest

from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.mcp_server import HOOK_TOOL_PREFIX, MCPHookServer
from sherma.hooks.types import BeforeLLMCallContext, NodeEnterContext


class _BareHooks(BaseHookExecutor):
    """An executor that overrides nothing — all base methods (pass-through)."""


class _PartialHooks(BaseHookExecutor):
    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        ctx.system_prompt = (ctx.system_prompt or "") + " [before]"
        return ctx

    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
        return None


class _RaisingHooks(BaseHookExecutor):
    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        raise RuntimeError("oops")


def test_bare_executor_registers_no_tools():
    server = MCPHookServer(executor=_BareHooks(), name="bare")
    assert server.registered_hooks == set()


def test_partial_executor_registers_only_overridden_hooks():
    server = MCPHookServer(executor=_PartialHooks(), name="partial")
    assert server.registered_hooks == {"before_llm_call", "node_enter"}


@pytest.mark.asyncio
async def test_registered_tool_names_use_prefix():
    server = MCPHookServer(executor=_PartialHooks(), name="partial")
    tools = await server.server.list_tools()
    names = {t.name for t in tools}
    assert f"{HOOK_TOOL_PREFIX}before_llm_call" in names
    assert f"{HOOK_TOOL_PREFIX}node_enter" in names


def _structured(result: object):
    if isinstance(result, tuple):
        _, structured = result
    else:
        structured = None
    return structured


@pytest.mark.asyncio
async def test_tool_invocation_delegates_to_executor():
    server = MCPHookServer(executor=_PartialHooks())
    tool_name = f"{HOOK_TOOL_PREFIX}before_llm_call"
    result = await server.server.call_tool(
        tool_name, {"params": {"system_prompt": "hi", "node_name": "n", "state": {}}}
    )
    structured = _structured(result)
    assert structured is not None
    inner = structured.get("result", structured)
    assert inner["system_prompt"] == "hi [before]"


@pytest.mark.asyncio
async def test_tool_invocation_can_return_none():
    server = MCPHookServer(executor=_PartialHooks())
    tool_name = f"{HOOK_TOOL_PREFIX}node_enter"
    result = await server.server.call_tool(
        tool_name,
        {"params": {"node_name": "x", "node_type": "call_llm", "state": {}}},
    )
    structured = _structured(result)
    if structured is not None:
        assert structured.get("result") is None


@pytest.mark.asyncio
async def test_executor_exception_surfaces_as_tool_error():
    """When a hook raises, the MCP server reports an error."""
    from mcp.server.fastmcp.exceptions import ToolError

    server = MCPHookServer(executor=_RaisingHooks())
    tool_name = f"{HOOK_TOOL_PREFIX}before_llm_call"
    with pytest.raises(ToolError):
        await server.server.call_tool(
            tool_name, {"params": {"system_prompt": "x", "node_name": "n", "state": {}}}
        )


def test_overriding_all_supported_hooks_registers_all():
    """Override every dispatchable hook and verify they're all exposed."""
    from sherma.hooks.apps.jsonrpc_handler import HOOK_METHODS

    class _AllHooks(BaseHookExecutor):
        pass

    # Dynamically override every wire-capable hook with a pass-through that
    # differs from the base method (so override-detection registers them).
    async def _passthrough(self, ctx):
        return ctx

    for name in HOOK_METHODS:
        setattr(_AllHooks, name, _passthrough)

    server = MCPHookServer(executor=_AllHooks())
    assert server.registered_hooks == set(HOOK_METHODS)
