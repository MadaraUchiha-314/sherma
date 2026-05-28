"""MCP server wrapper for HookHandler instances.

Exposes each overridden ``HookHandler`` method as an MCP tool named
``hooks.<hook_name>``.  The resulting :class:`mcp.server.fastmcp.FastMCP`
instance can be run with any supported transport (stdio, sse,
streamable-http) or mounted into an existing ASGI app.

Example::

    from sherma.hooks.handler import HookHandler
    from sherma.hooks.mcp_server import MCPHookServer

    class MyHooks(HookHandler):
        async def before_llm_call(self, params):
            params["system_prompt"] += "\\nBe concise."
            return params

    server = MCPHookServer(handler=MyHooks())
    server.run_stdio()
"""

from __future__ import annotations

import logging
from typing import Any

from sherma.hooks.apps.jsonrpc_handler import HOOK_METHODS
from sherma.hooks.handler import HookHandler

logger = logging.getLogger(__name__)

HOOK_TOOL_PREFIX = "hooks."


class MCPHookServer:
    """Wrap a :class:`HookHandler` as an MCP server.

    Only hook methods that the handler subclass actually overrides
    are exposed as MCP tools.  The base :class:`HookHandler` methods
    all return ``None`` (pass-through), so registering them would just
    add latency.

    The underlying :class:`mcp.server.fastmcp.FastMCP` instance is
    available via the :attr:`server` attribute for custom integration
    (mounting in an existing ASGI app, attaching middleware, etc.).
    """

    def __init__(
        self,
        handler: HookHandler,
        name: str = "sherma-hooks",
        instructions: str | None = None,
    ) -> None:
        from mcp.server.fastmcp import FastMCP

        self._handler = handler
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
            handler_method = getattr(self._handler, hook_name, None)
            base_method = getattr(HookHandler, hook_name, None)
            if handler_method is None or base_method is None:
                continue
            handler_func = getattr(handler_method, "__func__", handler_method)
            if handler_func is base_method:
                continue  # Not overridden; skip
            self._register_one(hook_name, handler_method)

    def _register_one(self, hook_name: str, method: Any) -> None:
        tool_name = f"{HOOK_TOOL_PREFIX}{hook_name}"

        async def _tool(params: dict[str, Any]) -> dict[str, Any] | None:
            try:
                return await method(params)
            except Exception:
                logger.exception("MCP hook tool '%s' raised an exception", tool_name)
                raise

        _tool.__name__ = hook_name
        _tool.__doc__ = method.__doc__ or f"sherma hook: {hook_name}"

        self._server.add_tool(
            _tool,
            name=tool_name,
            description=method.__doc__ or f"sherma hook: {hook_name}",
        )
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
