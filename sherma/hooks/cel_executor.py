"""CEL-expression based hook executor for declarative agents.

A :class:`CELHookExecutor` lets hook behaviour be defined entirely in the
declarative YAML: each hook type maps to a spec of CEL expressions that are
evaluated against the hook context. The result updates graph ``state`` (via
``state_updates``) and other modifiable context fields (via ``set``),
producing the same effect as the return value of a Python hook.

The same serialization rules that govern remote hooks decide which fields are
readable and which are writable, so the three transports stay consistent:

* non-serializable Python objects (``node_context``, ``agent``, ``chat_model``,
  ``registries``) and complex LangChain objects (``messages``, ``response``,
  ``tools``, ``tool_calls``) are **read-only** in CEL;
* ``error`` fields are read-only (exceptions cannot be produced from CEL);
* ``state`` is updated through ``state_updates`` (merge semantics);
* every other scalar/dict field is writable through ``set``;
* ``on_chat_model_create`` is unsupported (it returns a Python object).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

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
    """Return the context fields a CEL ``set`` block may assign for *hook_name*.

    Writable = all dataclass fields minus the non-serializable, complex, and
    error fields, minus ``state`` (which is updated via ``state_updates``).
    Returns an empty set for unknown / unsupported hook names.
    """
    ctx_class = _HOOK_CONTEXT_TYPES.get(hook_name)
    if ctx_class is None:
        return frozenset()
    excluded = _NON_SERIALIZABLE_FIELDS | _COMPLEX_FIELDS | _ERROR_FIELDS | {"state"}
    return frozenset(
        f.name for f in dataclasses.fields(ctx_class) if f.name not in excluded
    )


def hook_supports_state_updates(hook_name: str) -> bool:
    """Whether *hook_name* accepts ``state_updates``.

    True when the context exposes a ``result`` field (the node's returned
    state delta — these persist into the graph state) or a ``state`` field
    (in-flight state, visible to later hooks). False for unknown hooks.
    """
    ctx_class = _HOOK_CONTEXT_TYPES.get(hook_name)
    if ctx_class is None:
        return False
    names = {f.name for f in dataclasses.fields(ctx_class)}
    return "result" in names or "state" in names


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

    *spec* maps a hook name to its evaluation rules:

    ``{"when": <expr?>, "state_updates": {<key>: <expr>}, "set": {<field>: <expr>}}``

    Construct one per ``cel`` :class:`HookDef` entry. Hooks not present in
    *spec* are no-ops (inherited from :class:`BaseHookExecutor`).
    """

    def __init__(self, spec: dict[str, dict[str, Any]], cel: CelEngine) -> None:
        self._spec = spec
        self._cel = cel

    async def _apply(self, hook_name: str, ctx: Any) -> Any | None:
        rule = self._spec.get(hook_name)
        if rule is None:
            return None

        state = getattr(ctx, "state", None)
        state_dict: dict[str, Any] = state if isinstance(state, dict) else {}
        extra = _readable_extra(ctx)

        when = rule.get("when")
        if when is not None and not self._cel.evaluate_bool(when, state_dict, extra):
            return None

        state_updates: dict[str, str] = rule.get("state_updates") or {}
        if state_updates:
            # State updates merge into the dict the node returns as its state
            # delta. That is ``result`` for result-bearing contexts (node_exit,
            # node_execute, after_tool_call, ...); only those persist into the
            # graph state. Otherwise fall back to the in-flight ``state`` dict
            # (visible to later hooks).
            target = ctx.result if hasattr(ctx, "result") else ctx.state
            merged = dict(target)
            for key, expr in state_updates.items():
                merged[key] = self._cel.evaluate(expr, state_dict, extra=extra)
            if hasattr(ctx, "result"):
                ctx.result = merged
            else:
                ctx.state = merged

        set_fields: dict[str, str] = rule.get("set") or {}
        for field, expr in set_fields.items():
            setattr(ctx, field, self._cel.evaluate(expr, state_dict, extra=extra))

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
