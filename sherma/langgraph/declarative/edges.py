"""Edge builder functions for declarative agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END

from sherma.langgraph.declarative.cel_engine import CelEngine
from sherma.langgraph.declarative.nodes import INTERNAL_STATE_KEY
from sherma.langgraph.declarative.schema import EdgeDef
from sherma.langgraph.declarative.transform import HAS_ERROR_FALLBACK

# Built-in conditions that bypass CEL and use native Python checks.
_BUILTIN_CONDITIONS: dict[str, Callable[[dict[str, Any]], bool]] = {}


def _has_tool_calls(state: dict[str, Any]) -> bool:
    """Check if the last message in state has tool calls."""
    messages = state.get("messages", [])
    if not messages:
        return False
    last = messages[-1]
    # LangChain AIMessage has .tool_calls attribute
    tool_calls = getattr(last, "tool_calls", None)
    return bool(tool_calls)


def _has_error_fallback(state: dict[str, Any]) -> bool:
    """Check if the node set an error fallback sentinel in ``__sherma__``."""
    internal = state.get(INTERNAL_STATE_KEY, {})
    fallback = internal.get("error_fallback")
    if fallback:
        # Clear the sentinel so it doesn't trigger again on subsequent edges
        internal.pop("error_fallback", None)
        return True
    return False


_BUILTIN_CONDITIONS["has_tool_calls"] = _has_tool_calls
_BUILTIN_CONDITIONS[HAS_ERROR_FALLBACK] = _has_error_fallback


def _resolve_target(target: str) -> str:
    """Resolve special target names."""
    if target == "__end__":
        return END
    return target


def build_conditional_router(
    edge_def: EdgeDef,
    cel: CelEngine,
) -> tuple[Callable[..., str], dict[str, str]]:
    """Build a conditional router function and path map from an edge definition.

    Returns (router_fn, path_map) where path_map maps return values to node names.
    """
    branches = edge_def.branches or []
    default_target = _resolve_target(edge_def.default or END)

    path_map: dict[str, str] = {}
    for branch in branches:
        resolved = _resolve_target(branch.target)
        path_map[branch.target] = resolved
    if edge_def.default:
        path_map[edge_def.default] = default_target
    else:
        path_map["__end__"] = END

    def router(state: dict[str, Any]) -> str:
        for branch in branches:
            condition = branch.condition
            # Check built-in conditions first
            builtin = _BUILTIN_CONDITIONS.get(condition)
            if builtin is not None:
                if builtin(state):
                    return branch.target
            elif cel.evaluate_bool(condition, state):
                return branch.target
        return edge_def.default or "__end__"

    return router, path_map
