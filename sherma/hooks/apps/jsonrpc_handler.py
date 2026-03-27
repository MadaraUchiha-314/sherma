"""Shared JSON-RPC dispatch logic for hook servers."""

from __future__ import annotations

import logging
from typing import Any

from sherma.hooks.handler import HookHandler

logger = logging.getLogger(__name__)

# All hook method names that can be dispatched over JSON-RPC.
HOOK_METHODS: frozenset[str] = frozenset(
    {
        "before_llm_call",
        "after_llm_call",
        "before_tool_call",
        "after_tool_call",
        "before_agent_call",
        "after_agent_call",
        "before_skill_load",
        "after_skill_load",
        "node_enter",
        "node_execute",
        "node_exit",
        "before_interrupt",
        "after_interrupt",
        "before_graph_invoke",
        "after_graph_invoke",
        "on_node_error",
        "on_error",
    }
)


async def dispatch_jsonrpc(
    handler: HookHandler,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a JSON-RPC 2.0 request to the appropriate handler method.

    Returns a JSON-RPC 2.0 response dict.
    """
    method = body.get("method", "")
    params = body.get("params", {})
    request_id = body.get("id")

    if method not in HOOK_METHODS:
        return {"jsonrpc": "2.0", "result": None, "id": request_id}

    hook_fn = getattr(handler, method, None)
    if hook_fn is None:
        return {"jsonrpc": "2.0", "result": None, "id": request_id}

    try:
        result = await hook_fn(params)
    except Exception as exc:
        logger.exception("Hook handler '%s' raised an exception", method)
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": str(exc)},
            "id": request_id,
        }

    return {"jsonrpc": "2.0", "result": result, "id": request_id}
