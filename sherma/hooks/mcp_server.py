"""MCP server wrapper for hook executors.

Exposes each overridden hook of a :class:`~sherma.hooks.executor.HookExecutor`
as an MCP tool named ``hooks.<hook_name>``.  Hooks are authored exactly
like in-process hooks (subclass ``BaseHookExecutor`` and override the
typed context methods); this wrapper handles the (de)serialization at
the MCP boundary.

The resulting :class:`mcp.server.fastmcp.FastMCP` instance can be run
with any supported transport (stdio, sse, streamable-http) or mounted
into an existing ASGI app.

Example::

    from sherma.hooks.executor import BaseHookExecutor
    from sherma.hooks.mcp_server import MCPHookServer
    from sherma.hooks.types import BeforeLLMCallContext

    class MyHooks(BaseHookExecutor):
        async def before_llm_call(self, ctx: BeforeLLMCallContext):
            ctx.system_prompt += "\\nBe concise."
            return ctx

    server = MCPHookServer(executor=MyHooks())
    server.run_stdio()
"""

from __future__ import annotations

import logging
from typing import Any

from sherma.hooks.apps.jsonrpc_handler import HOOK_METHODS, run_executor_hook
from sherma.hooks.executor import BaseHookExecutor, HookExecutor

logger = logging.getLogger(__name__)

HOOK_TOOL_PREFIX = "hooks."


class MCPHookServer:
    """Wrap a :class:`HookExecutor` as an MCP server.

    Only hook methods the executor subclass actually overrides are
    exposed as MCP tools.  The base :class:`BaseHookExecutor` methods
    all return ``None`` (pass-through), so registering them would just
    add latency.

    The underlying :class:`mcp.server.fastmcp.FastMCP` instance is
    available via :attr:`server` for custom integration (mounting in an
    existing ASGI app, attaching middleware, etc.).
    """

    def __init__(
        self,
        executor: HookExecutor,
        name: str = "sherma-hooks",
        instructions: str | None = None,
    ) -> None:
        from mcp.server.fastmcp import FastMCP

        self._executor = executor
        self._server: FastMCP = FastMCP(name, instructions=instructions)
        self._registered_hooks: set[str] = set()
        self._register_tools()

    @property
    def server(self) -> Any:
        """The underlying ``FastMCP`` instance."""
        return self._server

    @property
    def registered_hooks(self) -> set[str]:
        """Names of hooks that were registered as MCP tools."""
        return set(self._registered_hooks)

    def _register_tools(self) -> None:
        for hook_name in sorted(HOOK_METHODS):
            impl_method = getattr(type(self._executor), hook_name, None)
            base_method = getattr(BaseHookExecutor, hook_name, None)
            if impl_method is None:
                continue
            if impl_method is base_method:
                continue  # Not overridden; skip
            self._register_one(hook_name)

    def _register_one(self, hook_name: str) -> None:
        tool_name = f"{HOOK_TOOL_PREFIX}{hook_name}"
        executor = self._executor
        doc = getattr(type(executor), hook_name).__doc__ or f"sherma hook: {hook_name}"

        async def _tool(params: dict[str, Any]) -> dict[str, Any] | None:
            try:
                return await run_executor_hook(executor, hook_name, params)
            except Exception:
                logger.exception("MCP hook tool '%s' raised an exception", tool_name)
                raise

        _tool.__name__ = hook_name
        _tool.__doc__ = doc

        self._server.add_tool(_tool, name=tool_name, description=doc)
        self._registered_hooks.add(hook_name)

    def run_stdio(self) -> None:
        """Run the MCP server over stdio (blocking)."""
        self._server.run(transport="stdio")

    def run_streamable_http(self, mount_path: str | None = None) -> None:
        """Run the MCP server over streamable HTTP (blocking)."""
        self._server.run(transport="streamable-http", mount_path=mount_path)

    def run_sse(self, mount_path: str | None = None) -> None:
        """Run the MCP server over SSE (blocking)."""
        self._server.run(transport="sse", mount_path=mount_path)
