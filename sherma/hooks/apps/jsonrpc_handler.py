"""Server-side hook dispatch shared by the JSON-RPC and MCP hook servers.

The server side runs a typed :class:`~sherma.hooks.executor.HookExecutor`
(authored exactly like an in-process hook).  Incoming params are
reconstructed into the matching context dataclass, the executor method
is invoked, and any returned context is serialized back to a dict.
"""

from __future__ import annotations

import logging
from typing import Any

from sherma.hooks.serialization import build_context_from_params, serialize_context

logger = logging.getLogger(__name__)

# All hook method names that can be dispatched over the wire
# (every hook except ``on_chat_model_create``, which returns a Python object).
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
        "before_skill_unload",
        "after_skill_unload",
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


async def run_executor_hook(
    executor: Any,
    hook_name: str,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """Run one hook on a typed ``HookExecutor`` from serialized *params*.

    Builds the typed context, invokes ``executor.<hook_name>(ctx)``, and
    returns the serialized modified context, or ``None`` for pass-through.
    """
    if hook_name not in HOOK_METHODS:
        return None
    method = getattr(executor, hook_name, None)
    if method is None:
        return None
    ctx = build_context_from_params(hook_name, params)
    if ctx is None:
        return None
    result_ctx = await method(ctx)
    if result_ctx is None:
        return None
    return serialize_context(result_ctx)


async def dispatch_jsonrpc(
    executor: Any,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a JSON-RPC 2.0 request to a typed ``HookExecutor``.

    Returns a JSON-RPC 2.0 response dict.
    """
    method = body.get("method", "")
    params = body.get("params", {})
    request_id = body.get("id")

    if method not in HOOK_METHODS:
        return {"jsonrpc": "2.0", "result": None, "id": request_id}

    try:
        result = await run_executor_hook(executor, method, params)
    except Exception as exc:
        logger.exception("Hook executor '%s' raised an exception", method)
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": str(exc)},
            "id": request_id,
        }

    return {"jsonrpc": "2.0", "result": result, "id": request_id}
