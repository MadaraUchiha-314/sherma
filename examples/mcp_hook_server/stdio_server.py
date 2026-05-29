"""Example MCP-based hook server (stdio transport).

Subclass ``BaseHookExecutor`` and override only the hooks you need --
the same typed context interface used by in-process hooks. Wrap it in
:class:`MCPHookServer` and run it as a stdio MCP server.

Run::

    python -m examples.mcp_hook_server.stdio_server

Then configure your agent to use this hook server:

    hooks:
      - mcp:
          command: python
          args: ["-m", "examples.mcp_hook_server.stdio_server"]
"""

from __future__ import annotations

from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.mcp_server import MCPHookServer
from sherma.hooks.types import (
    BeforeLLMCallContext,
    GraphInvokeContext,
    NodeEnterContext,
    NodeExitContext,
)


class MyHooks(BaseHookExecutor):
    """Example MCP hook executor demonstrating a few lifecycle hooks."""

    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        """Append a safety guardrail to the system prompt."""
        ctx.system_prompt += (
            "\n\nIMPORTANT: ALWAYS be helpful, accurate, "
            "and concise. Never fabricate data."
        )
        return ctx

    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
        """Logging hook: pass-through."""
        return None

    async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
        """Logging hook: pass-through."""
        return None

    async def before_graph_invoke(
        self, ctx: GraphInvokeContext
    ) -> GraphInvokeContext | None:
        """Tighten the recursion limit for safety."""
        ctx.config["recursion_limit"] = 50
        return ctx


def main() -> None:
    server = MCPHookServer(executor=MyHooks(), name="sherma-example-hooks")
    server.run_stdio()


if __name__ == "__main__":
    main()
