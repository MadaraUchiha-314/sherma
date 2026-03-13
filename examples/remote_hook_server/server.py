"""Example remote hook server using HookHandler + HookFastAPIApplication.

Subclass ``HookHandler`` and override only the hooks you need.
Pass it to ``HookFastAPIApplication`` to get a ready-to-run server.

Usage:
    pip install fastapi uvicorn
    uvicorn examples.remote_hook_server.server:app --port 8000

Then configure your agent to use this hook server:

    hooks:
      - url: http://localhost:8000/hooks
"""

from __future__ import annotations

from typing import Any

from sherma.hooks.apps import HookFastAPIApplication
from sherma.hooks.handler import HookHandler


class MyHooks(HookHandler):
    """Example hook handler demonstrating a few lifecycle hooks."""

    async def before_llm_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """Append a safety guardrail to the system prompt."""
        params["system_prompt"] += (
            "\n\nIMPORTANT: Always be helpful, accurate, "
            "and concise. Never fabricate data."
        )
        return params

    async def node_enter(self, params: dict[str, Any]) -> None:
        """Log when execution enters a node."""
        print(
            f"  [remote-hook] >>> Entering "
            f"'{params['node_name']}' "
            f"(type={params['node_type']})"
        )
        return None

    async def node_exit(self, params: dict[str, Any]) -> None:
        """Log when execution leaves a node."""
        print(f"  [remote-hook] <<< Exiting '{params['node_name']}'")
        return None

    async def before_graph_invoke(
        self, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Set a custom recursion limit and log the invocation."""
        print(
            f"  [remote-hook] Graph invoke: "
            f"agent={params['agent_id']}, "
            f"thread={params['thread_id']}"
        )
        params["config"]["recursion_limit"] = 50
        return params


app = HookFastAPIApplication(handler=MyHooks()).build(
    title="sherma remote hook server",
)
