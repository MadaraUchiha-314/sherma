"""Serialization utilities for hook context dataclasses over JSON-RPC."""

from __future__ import annotations

import dataclasses
from typing import Any

# Fields that are Python objects and cannot be serialized to JSON.
_NON_SERIALIZABLE_FIELDS: set[str] = {
    "node_context",
    "agent",
    "chat_model",
    "checkpointer",
    "registries",
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
