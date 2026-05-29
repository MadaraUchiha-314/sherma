"""Example remote hook server using BaseHookExecutor + HookFastAPIApplication.

Subclass ``BaseHookExecutor`` and override only the hooks you need --
the same typed context interface used by in-process hooks. Pass it to
``HookFastAPIApplication`` to get a ready-to-run JSON-RPC server.

Usage:
    pip install fastapi uvicorn
    uvicorn examples.remote_hook_server.server:app --port 8000

Then configure your agent to use this hook server:

    hooks:
      - url: http://localhost:8000/hooks
"""

from __future__ import annotations

from sherma.hooks.apps import HookFastAPIApplication
from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.types import (
    BeforeLLMCallContext,
    GraphInvokeContext,
    NodeEnterContext,
    NodeExitContext,
)


class MyHooks(BaseHookExecutor):
    """Example hook executor demonstrating a few lifecycle hooks."""

    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        """Append a safety guardrail to the system prompt."""
        ctx.system_prompt += (
            "\n\nIMPORTANT: Always be helpful, accurate, "
            "and concise. Never fabricate data."
        )
        return ctx

    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
        """Log when execution enters a node."""
        print(f"  [remote-hook] >>> Entering '{ctx.node_name}' (type={ctx.node_type})")
        return None

    async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
        """Log when execution leaves a node."""
        print(f"  [remote-hook] <<< Exiting '{ctx.node_name}'")
        return None

    async def before_graph_invoke(
        self, ctx: GraphInvokeContext
    ) -> GraphInvokeContext | None:
        """Set a custom recursion limit and log the invocation."""
        print(
            f"  [remote-hook] Graph invoke: "
            f"agent={ctx.agent_id}, thread={ctx.thread_id}"
        )
        ctx.config["recursion_limit"] = 50
        return ctx


app = HookFastAPIApplication(executor=MyHooks()).build(
    title="sherma remote hook server",
)
