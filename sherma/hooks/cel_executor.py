"""CEL-expression based hook executor for declarative agents.

A :class:`CELHookExecutor` lets hook behaviour be defined entirely in the
declarative YAML: each hook type maps to a single CEL expression that is
evaluated against the hook context and returns the hook's **return delta** — a
map of ``context_field -> new value``. Each returned key overwrites that field
on the context, producing the same effect as the return value of a Python
hook. An empty map (``{}``) is a no-op (pass-through), mirroring a Python hook
that returns ``None``.

The same serialization rules that govern remote hooks decide which fields are
writable, so the transports stay consistent:

* non-serializable Python objects (``node_context``, ``agent``, ``chat_model``,
  ``registries``) and complex LangChain objects (``messages``, ``response``,
  ``tools``, ``tool_calls``) cannot be set;
* ``error`` fields cannot be set (exceptions cannot be produced from CEL);
* every other field (``state``, ``result``, ``system_prompt``, ``config``, …)
  is writable;
* ``on_chat_model_create`` is unsupported (it returns a Python object).

Dict-valued fields (``state``, ``result``, ``config``, ``kwargs``, …) merge —
the returned sub-map adds/overwrites keys without discarding the rest, which
also preserves the original message objects already in ``result["messages"]``.
Other (scalar) fields are replaced.

Every context field is readable in the expression at the top level
(``node_name``, ``response``, ``result``, …); graph state is reached via
``state.<field>``. To persist a state change, return ``{"result": {...}}`` on a
hook whose context carries ``result`` (``node_exit``, ``node_execute``,
``after_*``) — that dict is the node's returned state delta. Mutating other
fields like ``state`` on a hook without ``result`` does not persist.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from sherma.exceptions import CelEvaluationError
from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.serialization import (
    _COMPLEX_FIELDS,
    _ERROR_FIELDS,
    _HOOK_CONTEXT_TYPES,
    _NON_SERIALIZABLE_FIELDS,
)

if TYPE_CHECKING:
    from sherma.langgraph.declarative.cel_engine import CelEngine

# Hook names a CEL hook may target. Mirrors the wire-capable hooks
# (``on_chat_model_create`` is intentionally excluded — it returns a Python
# chat model instance / factory that CEL cannot produce).
CEL_CAPABLE_HOOKS: frozenset[str] = frozenset(_HOOK_CONTEXT_TYPES)


def writable_fields(hook_name: str) -> frozenset[str]:
    """Return the context fields a CEL expression may overwrite for *hook_name*.

    Writable = all dataclass fields minus the non-serializable, complex, and
    error fields. Returns an empty set for unknown / unsupported hook names.
    """
    ctx_class = _HOOK_CONTEXT_TYPES.get(hook_name)
    if ctx_class is None:
        return frozenset()
    excluded = _NON_SERIALIZABLE_FIELDS | _COMPLEX_FIELDS | _ERROR_FIELDS
    return frozenset(
        f.name for f in dataclasses.fields(ctx_class) if f.name not in excluded
    )


def _readable_extra(ctx: Any) -> dict[str, Any]:
    """Build the top-level CEL variables from a hook context.

    Every field except ``state`` is exposed under its own name. ``state`` is
    passed separately so it is reachable as ``state.<field>`` (consistent with
    the rest of sherma's CEL surface).
    """
    extra: dict[str, Any] = {}
    for f in dataclasses.fields(ctx):
        if f.name == "state":
            continue
        extra[f.name] = getattr(ctx, f.name)
    return extra


class CELHookExecutor(BaseHookExecutor):
    """A hook executor whose behaviour is defined by CEL expressions.

    *expressions* maps a hook name to a single CEL expression that evaluates
    to a map of ``context_field -> new value`` (the hook's return delta).
    Construct one per ``cel`` hook entry. Hooks not present in *expressions*
    are no-ops (inherited from :class:`BaseHookExecutor`).
    """

    def __init__(self, expressions: dict[str, str], cel: CelEngine) -> None:
        self._expressions = expressions
        self._cel = cel

    async def _apply(self, hook_name: str, ctx: Any) -> Any | None:
        expr = self._expressions.get(hook_name)
        if expr is None:
            return None

        state = getattr(ctx, "state", None)
        state_dict: dict[str, Any] = state if isinstance(state, dict) else {}
        extra = _readable_extra(ctx)

        delta = self._cel.evaluate(expr, state_dict, extra=extra)
        if not delta:
            return None  # empty map / falsy -> no-op (pass through)
        if not isinstance(delta, dict):
            raise CelEvaluationError(
                f"CEL hook '{hook_name}' must evaluate to a map of "
                f"field -> value, got {type(delta).__name__}"
            )

        allowed = writable_fields(hook_name)
        for field, value in delta.items():
            if field not in allowed:
                raise CelEvaluationError(
                    f"CEL hook '{hook_name}' cannot set field '{field}'; "
                    f"writable fields are {sorted(allowed)}."
                )
            current = getattr(ctx, field, None)
            if isinstance(current, dict) and isinstance(value, dict):
                # Dict fields (state, result, config, kwargs, ...) merge so the
                # returned delta adds/overwrites keys without discarding others
                # — notably preserving the original message objects already in
                # ``result["messages"]`` rather than the CEL-converted copies.
                setattr(ctx, field, {**current, **value})
            else:
                setattr(ctx, field, value)
        return ctx

    async def before_llm_call(self, ctx: Any) -> Any | None:
        return await self._apply("before_llm_call", ctx)

    async def after_llm_call(self, ctx: Any) -> Any | None:
        return await self._apply("after_llm_call", ctx)

    async def before_tool_call(self, ctx: Any) -> Any | None:
        return await self._apply("before_tool_call", ctx)

    async def after_tool_call(self, ctx: Any) -> Any | None:
        return await self._apply("after_tool_call", ctx)

    async def before_agent_call(self, ctx: Any) -> Any | None:
        return await self._apply("before_agent_call", ctx)

    async def after_agent_call(self, ctx: Any) -> Any | None:
        return await self._apply("after_agent_call", ctx)

    async def before_skill_load(self, ctx: Any) -> Any | None:
        return await self._apply("before_skill_load", ctx)

    async def after_skill_load(self, ctx: Any) -> Any | None:
        return await self._apply("after_skill_load", ctx)

    async def before_skill_unload(self, ctx: Any) -> Any | None:
        return await self._apply("before_skill_unload", ctx)

    async def after_skill_unload(self, ctx: Any) -> Any | None:
        return await self._apply("after_skill_unload", ctx)

    async def node_enter(self, ctx: Any) -> Any | None:
        return await self._apply("node_enter", ctx)

    async def node_execute(self, ctx: Any) -> Any | None:
        return await self._apply("node_execute", ctx)

    async def node_exit(self, ctx: Any) -> Any | None:
        return await self._apply("node_exit", ctx)

    async def before_interrupt(self, ctx: Any) -> Any | None:
        return await self._apply("before_interrupt", ctx)

    async def after_interrupt(self, ctx: Any) -> Any | None:
        return await self._apply("after_interrupt", ctx)

    async def before_graph_invoke(self, ctx: Any) -> Any | None:
        return await self._apply("before_graph_invoke", ctx)

    async def after_graph_invoke(self, ctx: Any) -> Any | None:
        return await self._apply("after_graph_invoke", ctx)

    async def on_node_error(self, ctx: Any) -> Any | None:
        return await self._apply("on_node_error", ctx)

    async def on_error(self, ctx: Any) -> Any | None:
        return await self._apply("on_error", ctx)
