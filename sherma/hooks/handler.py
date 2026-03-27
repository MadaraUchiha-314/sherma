"""HookHandler: interface for implementing remote hook servers.

Subclass ``HookHandler`` and override only the hooks you need.
Pass the instance to :class:`HookStarletteApplication` or
:class:`HookFastAPIApplication` to get a ready-to-run ASGI server.

Example::

    from sherma.hooks.handler import HookHandler

    class MyHooks(HookHandler):
        async def before_llm_call(self, params):
            params["system_prompt"] += "\\nBe concise."
            return params
"""

from __future__ import annotations

from typing import Any


class HookHandler:
    """Base class for remote hook implementations.

    Each method corresponds to a lifecycle hook point.  Return
    ``None`` to pass the context through unchanged, or return a
    modified *params* dict to replace it for subsequent hooks.

    Override only the hooks you care about -- all methods default to
    ``None`` (pass-through).

    The ``on_chat_model_create`` hook is intentionally absent because
    it requires returning a Python object and cannot work over
    JSON-RPC.
    """

    async def before_llm_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def after_llm_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def before_tool_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def after_tool_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def before_agent_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def after_agent_call(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def before_skill_load(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def after_skill_load(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def node_enter(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def node_execute(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def node_exit(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def before_interrupt(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def after_interrupt(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def before_graph_invoke(
        self, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        return None

    async def after_graph_invoke(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def on_node_error(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None

    async def on_error(self, params: dict[str, Any]) -> dict[str, Any] | None:
        return None
