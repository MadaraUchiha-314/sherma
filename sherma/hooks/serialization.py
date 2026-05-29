"""Serialization utilities for hook context dataclasses over the wire.

Used by both transports (JSON-RPC and MCP).  The agent (client) side
serializes a context dataclass to a JSON-safe dict via
:func:`serialize_context` and reconstructs the modified context with
:func:`deserialize_into_context`.  The hook *server* side rebuilds a
typed context dataclass from the incoming params dict via
:func:`build_context_from_params`, so remote hooks are authored with
the same typed contexts as in-process hooks.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from sherma.hooks.types import (
    AfterAgentCallContext,
    AfterGraphInvokeContext,
    AfterInterruptContext,
    AfterLLMCallContext,
    AfterSkillLoadContext,
    AfterSkillUnloadContext,
    AfterToolCallContext,
    BeforeAgentCallContext,
    BeforeInterruptContext,
    BeforeLLMCallContext,
    BeforeSkillLoadContext,
    BeforeSkillUnloadContext,
    BeforeToolCallContext,
    GraphInvokeContext,
    NodeEnterContext,
    NodeExecuteContext,
    NodeExitContext,
    OnErrorContext,
    OnNodeErrorContext,
)

# Fields that are Python objects and cannot be serialized to JSON.
_NON_SERIALIZABLE_FIELDS: set[str] = {
    "node_context",
    "agent",
    "chat_model",
    "registries",
}

# Maps a hook name to the context dataclass the server side reconstructs.
# ``on_chat_model_create`` is intentionally absent (cannot cross the wire).
_HOOK_CONTEXT_TYPES: dict[str, type] = {
    "before_llm_call": BeforeLLMCallContext,
    "after_llm_call": AfterLLMCallContext,
    "before_tool_call": BeforeToolCallContext,
    "after_tool_call": AfterToolCallContext,
    "before_agent_call": BeforeAgentCallContext,
    "after_agent_call": AfterAgentCallContext,
    "before_skill_load": BeforeSkillLoadContext,
    "after_skill_load": AfterSkillLoadContext,
    "before_skill_unload": BeforeSkillUnloadContext,
    "after_skill_unload": AfterSkillUnloadContext,
    "node_enter": NodeEnterContext,
    "node_execute": NodeExecuteContext,
    "node_exit": NodeExitContext,
    "before_interrupt": BeforeInterruptContext,
    "after_interrupt": AfterInterruptContext,
    "before_graph_invoke": GraphInvokeContext,
    "after_graph_invoke": AfterGraphInvokeContext,
    "on_node_error": OnNodeErrorContext,
    "on_error": OnErrorContext,
}

# Fields containing BaseException instances.
_ERROR_FIELDS: set[str] = {"error"}

# Fields containing complex LangChain objects (read-only for remote hooks).
_COMPLEX_FIELDS: set[str] = {"messages", "response", "tools", "tool_calls"}


def _serialize_error(error: BaseException | None) -> dict[str, str] | None:
    if error is None:
        return None
    return {"type": type(error).__name__, "message": str(error)}


def _serialize_complex(value: Any) -> Any:
    """Best-effort serialization of LangChain objects for observation."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_serialize_complex(item) for item in value]
    if isinstance(value, dict):
        return value
    for method in ("model_dump", "dict"):
        fn = getattr(value, method, None)
        if fn is not None:
            try:
                result = fn()
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
    return str(value)


def serialize_context(ctx: Any) -> dict[str, Any]:
    """Convert a hook context dataclass to a JSON-safe dict.

    Non-serializable fields (``node_context``, ``agent``, ``chat_model``)
    are omitted.  Error fields are converted to ``{"type": ..., "message": ...}``.
    Complex LangChain objects are serialized best-effort for observation.
    """
    result: dict[str, Any] = {}
    for f in dataclasses.fields(ctx):
        name = f.name
        value = getattr(ctx, name)
        if name in _NON_SERIALIZABLE_FIELDS:
            continue
        if name in _ERROR_FIELDS:
            result[name] = _serialize_error(value)
        elif name in _COMPLEX_FIELDS:
            result[name] = _serialize_complex(value)
        else:
            result[name] = value
    return result


def _rebuild_error(data: Any) -> BaseException | None:
    """Reconstruct a stand-in exception from a serialized ``{type, message}``.

    The original exception cannot cross the wire, so a fresh exception
    class with the original type *name* is synthesised carrying the
    message.  This lets server-side hooks inspect ``type(ctx.error).__name__``
    and ``str(ctx.error)``.
    """
    if not data or not isinstance(data, dict):
        return None
    name = data.get("type") or "RemoteHookError"
    message = data.get("message", "")
    try:
        exc_cls = type(str(name), (Exception,), {})
        return exc_cls(message)
    except Exception:
        return Exception(message)


def build_context_from_params(hook_name: str, params: dict[str, Any]) -> Any:
    """Reconstruct a typed hook context dataclass from a params dict.

    Used on the hook *server* side so that handlers receive the same
    typed context objects as in-process hooks.  Non-serializable fields
    (``node_context``, ``agent``, ``registries``) are set to ``None``;
    error fields are rebuilt as stand-in exceptions; complex fields
    (``messages``, ``tools``, ...) carry their serialized form.

    Returns ``None`` if *hook_name* is not a wire-capable hook.
    """
    ctx_class = _HOOK_CONTEXT_TYPES.get(hook_name)
    if ctx_class is None:
        return None

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(ctx_class):
        name = f.name
        if name in _NON_SERIALIZABLE_FIELDS:
            kwargs[name] = None
        elif name in _ERROR_FIELDS:
            kwargs[name] = _rebuild_error(params.get(name))
        elif name in params:
            kwargs[name] = params[name]
        elif (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ):
            # Required field absent from the payload — supply a safe default.
            kwargs[name] = None
        # else: a default / default_factory exists; let the dataclass apply it.
    return ctx_class(**kwargs)


def deserialize_into_context(
    ctx_class: type,
    data: dict[str, Any],
    original_ctx: Any,
) -> Any:
    """Reconstruct a context dataclass from a JSON-RPC response.

    Non-serializable and complex fields are re-attached from
    *original_ctx*.  Primitive and ``state`` fields are taken from
    *data* when present.
    """
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(original_ctx):
        name = f.name
        if name in _NON_SERIALIZABLE_FIELDS:
            kwargs[name] = getattr(original_ctx, name)
        elif name in _ERROR_FIELDS:
            kwargs[name] = getattr(original_ctx, name)
        elif name in _COMPLEX_FIELDS:
            kwargs[name] = getattr(original_ctx, name)
        elif name in data:
            kwargs[name] = data[name]
        else:
            kwargs[name] = getattr(original_ctx, name)
    return ctx_class(**kwargs)
