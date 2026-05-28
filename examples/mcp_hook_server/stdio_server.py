"""Example MCP-based hook server (stdio transport).

Subclass ``HookHandler`` and override only the hooks you need, then
wrap the handler in :class:`MCPHookServer` and run it as a stdio MCP
server.

Run::

    python -m examples.mcp_hook_server.stdio_server

Then configure your agent to use this hook server:

    hooks:
      - mcp:
          command: python
          args: ["-m", "examples.mcp_hook_server.stdio_server"]
"""

from __future__ import annotations

from typing import Any

from sherma.hooks.handler import HookHandler
from sherma.hooks.mcp_server import MCPHookServer


class MyHooks(HookHandler):
    """Example MCP hook handler demonstrating a few lifecycle hooks."""

    async def before_llm_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """Append a safety guardrail to the system prompt."""
        params["system_prompt"] = (
            params.get("system_prompt", "")
            + "\n\nIMPORTANT: ALWAYS be helpful, accurate, "
            "and concise. Never fabricate data."
        )
        return params

    async def node_enter(self, params: dict[str, Any]) -> None:
        """Logging hook: pass-through."""
        return None

    async def node_exit(self, params: dict[str, Any]) -> None:
        """Logging hook: pass-through."""
        return None

    async def before_graph_invoke(
        self, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Tighten the recursion limit for safety."""
        config = params.get("config") or {}
        config["recursion_limit"] = 50
        params["config"] = config
        return params


def main() -> None:
    server = MCPHookServer(handler=MyHooks(), name="sherma-example-hooks")
    server.run_stdio()


if __name__ == "__main__":
    main()
