"""Node factory functions for declarative agents."""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from sherma.hooks.manager import HookManager
from sherma.hooks.types import OnNodeErrorContext
from sherma.langgraph.declarative.cel_engine import CelEngine
from sherma.langgraph.declarative.schema import RegistryRef
from sherma.langgraph.tools import to_langgraph_tool
from sherma.logging import get_logger

_BARE_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")

# Patterns that match simple ``state.key`` or ``state["key"]`` CEL expressions
# so we can short-circuit CEL evaluation and preserve LangChain message objects.
_STATE_DOT_RE = re.compile(r"^state\.(\w+)$")
_STATE_BRACKET_RE = re.compile(r'^state\["(\w+)"\]$')

INTERNAL_STATE_KEY = "__sherma__"
"""Top-level state key for all sherma-managed internal data."""

logger = get_logger(__name__)


def _extract_state_key(expr: str) -> str | None:
    """Extract the state field name from ``state.key`` or ``state["key"]``.

    Returns the key string if *expr* is a simple state access, else ``None``.
    """
    m = _STATE_DOT_RE.match(expr) or _STATE_BRACKET_RE.match(expr)
    return m.group(1) if m else None


def _get_internal(state: dict[str, Any]) -> dict[str, Any]:
    """Return the internal sherma state dict (never mutates *state*)."""
    return dict(state.get(INTERNAL_STATE_KEY, {}))


def _set_internal(result: dict[str, Any], internal: dict[str, Any]) -> None:
    """Write the internal sherma state dict into a result dict."""
    result[INTERNAL_STATE_KEY] = internal


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from sherma.langgraph.declarative.schema import (
        CallAgentArgs,
        CallLLMArgs,
        DataTransformArgs,
        DeclarativeConfig,
        InterruptArgs,
        LoadSkillsArgs,
        NodeDef,
        RetryPolicy,
        SetStateArgs,
        ToolNodeArgs,
    )
    from sherma.registry.skill import SkillRegistry
    from sherma.registry.tool import ToolRegistry


@dataclass
class NodeContext:
    """Dependencies injected into every node at build time.

    Every node function receives a ``NodeContext`` as its first argument
    (bound via ``functools.partial``).  This gives nodes access to the
    full declarative config and their own node definition without
    polluting the LangGraph state.
    """

    config: DeclarativeConfig
    node_def: NodeDef
    extra: dict[str, Any] = field(default_factory=dict)
    hook_manager: HookManager | None = None


async def _run_node_error_hook(
    hooks: HookManager | None,
    ctx: NodeContext,
    state: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    """Run on_node_error hook chain. Returns fallback if consumed, else re-raises."""
    if isinstance(exc, GraphBubbleUp):
        raise exc  # never intercept interrupt flow
    if hooks is None:
        raise exc
    err_ctx = await hooks.run_hook(
        "on_node_error",
        OnNodeErrorContext(
            node_context=ctx,
            node_name=ctx.node_def.name,
            node_type=ctx.node_def.type,
            error=exc,
            state=state,
        ),
    )
    if err_ctx.error is None:
        return {}
    raise err_ctx.error from exc


def _compute_delay(retry: RetryPolicy, attempt: int) -> float:
    """Compute the delay in seconds before the next retry attempt."""
    if retry.strategy == "fixed":
        return min(retry.delay, retry.max_delay)
    # exponential: delay * 2^(attempt-1)
    return min(retry.delay * (2 ** (attempt - 1)), retry.max_delay)


def _store_error_and_fallback(
    state: dict[str, Any],
    node_name: str,
    exc: Exception,
    attempt: int,
    fallback: str,
) -> dict[str, Any]:
    """Store error info in ``__sherma__`` and set the fallback sentinel."""
    internal = _get_internal(state)
    internal["last_error"] = {
        "node": node_name,
        "type": type(exc).__qualname__,
        "message": str(exc),
        "attempt": attempt,
    }
    internal["error_fallback"] = fallback
    result: dict[str, Any] = {}
    _set_internal(result, internal)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _resolve_all_registry_tools(
    tool_registry: ToolRegistry,
) -> list[BaseTool]:
    """Return all tools currently in the ToolRegistry as LangChain tools."""
    resolved: list[BaseTool] = []
    for versions in tool_registry._entries.values():
        for entry in versions.values():
            tool_entity = await tool_registry._resolve(entry)
            resolved.append(to_langgraph_tool(tool_entity))
    return resolved


async def _resolve_skill_tools_from_state(
    state: dict[str, Any],
    tool_registry: ToolRegistry,
) -> list[BaseTool]:
    """Resolve skill tools from internal state ``loaded_tools_from_skills``."""
    internal = _get_internal(state)
    skill_tool_ids: list[str] = internal.get("loaded_tools_from_skills", [])
    resolved: list[BaseTool] = []
    for tool_id in skill_tool_ids:
        try:
            tool_entity = await tool_registry.get(tool_id, "*")
            resolved.append(to_langgraph_tool(tool_entity))
        except Exception:
            logger.warning("Could not resolve skill tool '%s' from registry", tool_id)
    return resolved


# ---------------------------------------------------------------------------
# call_llm
# ---------------------------------------------------------------------------


def build_call_llm_node(
    ctx: NodeContext,
    chat_model: BaseChatModel,
    cel: CelEngine,
    tool_registry: ToolRegistry | None = None,
) -> Callable[..., Any]:
    """Build a call_llm node function.

    Tool binding mode is determined by the node's own config
    (``ctx.node_def.args``):

    * ``use_tools_from_loaded_skills`` → binds only tools whose IDs
      appear in internal state ``loaded_tools_from_skills``.
    * ``use_tools_from_registry`` → binds all tools in the registry.
    * ``tools`` (explicit list) → resolves those specific tools from
      the registry at invocation time.
    * None of the above → no tools bound.
    """
    args: CallLLMArgs = ctx.node_def.args  # type: ignore[assignment]

    async def call_llm_fn(_ctx: NodeContext, state: dict[str, Any]) -> dict[str, Any]:
        hooks = _ctx.hook_manager

        try:
            # node_enter
            if hooks:
                from sherma.hooks.types import NodeEnterContext

                await hooks.run_hook(
                    "node_enter",
                    NodeEnterContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        state=state,
                    ),
                )

            current_tools: list[BaseTool] = []
            if args.use_tools_from_loaded_skills and tool_registry is not None:
                current_tools = await _resolve_skill_tools_from_state(
                    state, tool_registry
                )
            elif args.use_tools_from_registry and tool_registry is not None:
                current_tools = await _resolve_all_registry_tools(tool_registry)
            elif args.use_sub_agents_as_tools and tool_registry is not None:
                if args.use_sub_agents_as_tools == "all":
                    sub_agent_tool_ids: list[str] = _ctx.extra.get(
                        "sub_agent_tool_ids", []
                    )
                    if sub_agent_tool_ids:
                        refs = [RegistryRef(id=tid) for tid in sub_agent_tool_ids]
                        current_tools = await resolve_tools_for_node_async(
                            refs, tool_registry
                        )
                else:
                    # list[RegistryRef] — resolve specific sub-agents
                    current_tools = await resolve_tools_for_node_async(
                        args.use_sub_agents_as_tools, tool_registry
                    )

            # Merge explicit tools (additive with any dynamic flag)
            if args.tools and tool_registry is not None:
                explicit_tools = await resolve_tools_for_node_async(
                    args.tools, tool_registry
                )
                existing_names = {t.name for t in current_tools}
                for tool in explicit_tools:
                    if tool.name not in existing_names:
                        current_tools.append(tool)
                        existing_names.add(tool.name)

            # Build messages from array prompt
            role_map = {
                "system": SystemMessage,
                "human": HumanMessage,
                "ai": AIMessage,
            }
            all_messages: list[Any] = []
            system_parts: list[str] = []
            for item in args.prompt:
                if item.role == "messages":
                    # Use raw state value for simple state.key expressions
                    # to preserve LangChain message objects; fall back to
                    # CEL eval for complex expressions.
                    raw_key = _extract_state_key(item.content)
                    if (
                        raw_key is not None
                        and raw_key in state
                        and isinstance(state[raw_key], list)
                    ):
                        all_messages.extend(state[raw_key])
                    else:
                        all_messages.extend(cel.evaluate(item.content, state))
                else:
                    evaluated = cel.evaluate(item.content, state)
                    msg = role_map[item.role](content=str(evaluated))
                    all_messages.append(msg)
                    if item.role == "system":
                        system_parts.append(str(evaluated))

            system_prompt = "\n".join(system_parts)
            non_system_messages = [
                m for m in all_messages if not isinstance(m, SystemMessage)
            ]

            # before_llm_call
            if hooks:
                from sherma.hooks.types import BeforeLLMCallContext

                before_ctx = await hooks.run_hook(
                    "before_llm_call",
                    BeforeLLMCallContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        messages=non_system_messages,
                        system_prompt=system_prompt,
                        tools=current_tools,
                        state=state,
                    ),
                )
                # Rebuild all_messages from hook output
                hook_system = before_ctx.system_prompt
                hook_messages = before_ctx.messages
                current_tools = before_ctx.tools
                all_messages = []
                if hook_system:
                    all_messages.append(SystemMessage(content=hook_system))
                all_messages.extend(hook_messages)

            model: Any = chat_model
            if current_tools:
                model = model.bind_tools(current_tools)

            if args.response_format:
                json_schema = {
                    "name": args.response_format.name,
                    "description": args.response_format.description,
                    "schema": args.response_format.schema_,
                }
                model = model.with_structured_output(json_schema)

            logger.info(
                "[%s] Invoking LLM (%d tools) with %d messages,"
                " system prompt: %.100s...",
                _ctx.node_def.name,
                len(current_tools),
                len(all_messages),
                system_prompt,
            )

            # Retry loop wraps only model.ainvoke() — the IO call.
            on_error = _ctx.node_def.on_error
            retry = on_error.retry if on_error else None
            max_attempts = retry.max_attempts if retry else 1

            response: Any = None
            last_invoke_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    response = await model.ainvoke(all_messages)
                    last_invoke_exc = None
                    break
                except Exception as invoke_exc:
                    if isinstance(invoke_exc, GraphBubbleUp):
                        raise
                    last_invoke_exc = invoke_exc
                    logger.warning(
                        "[%s] LLM attempt %d/%d failed: %s",
                        _ctx.node_def.name,
                        attempt,
                        max_attempts,
                        invoke_exc,
                    )
                    if attempt < max_attempts and retry:
                        delay = _compute_delay(retry, attempt)
                        await asyncio.sleep(delay)

            if last_invoke_exc is not None:
                raise last_invoke_exc

            if args.response_format and isinstance(response, dict):
                import json

                response = AIMessage(content=json.dumps(response))

            # after_llm_call
            if hooks:
                from sherma.hooks.types import AfterLLMCallContext

                after_ctx = await hooks.run_hook(
                    "after_llm_call",
                    AfterLLMCallContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        response=response,
                        state=state,
                    ),
                )
                response = after_ctx.response

            content = getattr(response, "content", "")
            tool_calls = getattr(response, "tool_calls", [])
            logger.info(
                "[%s] LLM response: %.200s... | tool_calls=%d",
                _ctx.node_def.name,
                str(content),
                len(tool_calls) if tool_calls else 0,
            )
            if tool_calls:
                for tc in tool_calls:
                    logger.info(
                        "[%s]   tool_call: %s(%s)",
                        _ctx.node_def.name,
                        tc.get("name", "?"),
                        tc.get("args", {}),
                    )
            result: dict[str, Any] = {"messages": [response]}

            # node_exit
            if hooks:
                from sherma.hooks.types import NodeExitContext

                exit_ctx = await hooks.run_hook(
                    "node_exit",
                    NodeExitContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        result=result,
                        state=state,
                    ),
                )
                result = exit_ctx.result

            return result
        except Exception as exc:
            if isinstance(exc, GraphBubbleUp):
                raise
            on_error = _ctx.node_def.on_error
            if on_error and on_error.fallback:
                _retry = on_error.retry
                _max = _retry.max_attempts if _retry else 1
                return _store_error_and_fallback(
                    state,
                    _ctx.node_def.name,
                    exc,
                    attempt=_max,
                    fallback=on_error.fallback,
                )
            return await _run_node_error_hook(hooks, _ctx, state, exc)

    return partial(call_llm_fn, ctx)


# ---------------------------------------------------------------------------
# tool_node
# ---------------------------------------------------------------------------


def build_tool_node(
    ctx: NodeContext,
    tool_registry: ToolRegistry | None = None,
    skill_registry: SkillRegistry | None = None,
) -> Callable[..., Any]:
    """Build a tool_node that executes tool calls from the last AIMessage.

    Tool resolution is determined by the node's own config
    (``ctx.node_def.args``):

    * ``args.tools`` (explicit list) → resolves those specific tools
      from the registry.
    * No explicit tools → resolves all tools from the registry.

    When *skill_registry* is provided the node additionally
    inspects ``load_skill_md`` tool calls and updates the internal
    ``loaded_tools_from_skills`` list so that downstream
    ``use_tools_from_loaded_skills`` LLM nodes can pick them up.
    """
    if tool_registry is None:
        raise ValueError("tool_node requires a tool_registry")

    args: ToolNodeArgs = ctx.node_def.args  # type: ignore[assignment]

    async def tool_node_fn(_ctx: NodeContext, state: dict[str, Any]) -> dict[str, Any]:
        hooks = _ctx.hook_manager

        try:
            # node_enter
            if hooks:
                from sherma.hooks.types import NodeEnterContext

                await hooks.run_hook(
                    "node_enter",
                    NodeEnterContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        state=state,
                    ),
                )

            if args.tools:
                current_tools = await resolve_tools_for_node_async(
                    args.tools, tool_registry
                )
            else:
                current_tools = await _resolve_all_registry_tools(tool_registry)

            if not current_tools:
                logger.warning(
                    "[%s] tool_node invoked but no tools resolved",
                    _ctx.node_def.name,
                )
                return {"messages": []}

            # before_tool_call
            messages = state.get("messages", [])
            last_msg = messages[-1] if messages else None
            pending_calls: list[dict[str, Any]] = getattr(last_msg, "tool_calls", [])

            if hooks:
                from sherma.hooks.types import BeforeToolCallContext

                before_ctx = await hooks.run_hook(
                    "before_tool_call",
                    BeforeToolCallContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        tool_calls=pending_calls,
                        tools=current_tools,
                        state=state,
                    ),
                )
                current_tools = before_ctx.tools

            tool_node = ToolNode(current_tools)
            result: dict[str, Any] = await tool_node.ainvoke(state)

            # after_tool_call
            if hooks:
                from sherma.hooks.types import AfterToolCallContext

                after_ctx = await hooks.run_hook(
                    "after_tool_call",
                    AfterToolCallContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        result=result,
                        state=state,
                    ),
                )
                result = after_ctx.result

            # Track skill tool IDs when load_skill_md is called.
            if skill_registry is not None:
                internal = _get_internal(state)
                current_ids: list[str] = list(
                    internal.get("loaded_tools_from_skills", [])
                )
                for tc in pending_calls:
                    if tc.get("name") != "load_skill_md":
                        continue
                    skill_id = tc.get("args", {}).get("skill_id", "")
                    version = tc.get("args", {}).get("version", "*")
                    if not skill_id:
                        continue
                    if _BARE_VERSION_RE.match(version):
                        version = f"=={version}"
                    try:
                        skill = await skill_registry.get(skill_id, version)
                        card = skill.skill_card
                        if card:
                            for tool_id in card.local_tools:
                                if tool_id not in current_ids:
                                    current_ids.append(tool_id)
                            for mcp_id in card.mcps:
                                if mcp_id not in current_ids:
                                    current_ids.append(mcp_id)
                    except Exception:
                        logger.warning(
                            "[%s] Could not resolve skill for '%s'",
                            _ctx.node_def.name,
                            skill_id,
                        )
                internal["loaded_tools_from_skills"] = current_ids
                _set_internal(result, internal)

            # node_exit
            if hooks:
                from sherma.hooks.types import NodeExitContext

                exit_ctx = await hooks.run_hook(
                    "node_exit",
                    NodeExitContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        result=result,
                        state=state,
                    ),
                )
                result = exit_ctx.result

            return result
        except Exception as exc:
            if isinstance(exc, GraphBubbleUp):
                raise
            on_error = _ctx.node_def.on_error
            if on_error and on_error.fallback:
                return _store_error_and_fallback(
                    state,
                    _ctx.node_def.name,
                    exc,
                    attempt=1,
                    fallback=on_error.fallback,
                )
            return await _run_node_error_hook(hooks, _ctx, state, exc)

    return partial(tool_node_fn, ctx)


# ---------------------------------------------------------------------------
# call_agent
# ---------------------------------------------------------------------------


def build_call_agent_node(
    ctx: NodeContext,
    agent: Any,
    cel: CelEngine,
) -> Callable[..., Any]:
    """Build a call_agent node function."""
    args: CallAgentArgs = ctx.node_def.args  # type: ignore[assignment]

    async def call_agent_fn(_ctx: NodeContext, state: dict[str, Any]) -> dict[str, Any]:
        hooks = _ctx.hook_manager

        try:
            # node_enter
            if hooks:
                from sherma.hooks.types import NodeEnterContext

                await hooks.run_hook(
                    "node_enter",
                    NodeEnterContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        state=state,
                    ),
                )

            input_val = cel.evaluate(args.input, state)

            # before_agent_call
            if hooks:
                from sherma.hooks.types import BeforeAgentCallContext

                before_ctx = await hooks.run_hook(
                    "before_agent_call",
                    BeforeAgentCallContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        input_value=input_val,
                        agent=agent,
                        state=state,
                    ),
                )
                input_val = before_ctx.input_value

            from a2a.types import Message as A2AMessage
            from a2a.types import Part, Role, TextPart

            msg = A2AMessage(
                message_id="declarative-call",
                parts=[Part(root=TextPart(text=str(input_val)))],
                role=Role.user,
            )
            results: list[Any] = []
            async for event in agent.send_message(msg):
                results.append(event)

            result: dict[str, Any] = {}
            if results:
                last = results[-1]
                if isinstance(last, A2AMessage):
                    text_parts = [
                        p.root.text for p in last.parts if p.root.kind == "text"
                    ]
                    content = " ".join(text_parts)
                    result = {"messages": [AIMessage(content=content)]}

            # after_agent_call
            if hooks:
                from sherma.hooks.types import AfterAgentCallContext

                after_ctx = await hooks.run_hook(
                    "after_agent_call",
                    AfterAgentCallContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        result=result,
                        state=state,
                    ),
                )
                result = after_ctx.result

            # node_exit
            if hooks:
                from sherma.hooks.types import NodeExitContext

                exit_ctx = await hooks.run_hook(
                    "node_exit",
                    NodeExitContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        result=result,
                        state=state,
                    ),
                )
                result = exit_ctx.result

            return result
        except Exception as exc:
            if isinstance(exc, GraphBubbleUp):
                raise
            on_error = _ctx.node_def.on_error
            if on_error and on_error.fallback:
                return _store_error_and_fallback(
                    state,
                    _ctx.node_def.name,
                    exc,
                    attempt=1,
                    fallback=on_error.fallback,
                )
            return await _run_node_error_hook(hooks, _ctx, state, exc)

    return partial(call_agent_fn, ctx)


# ---------------------------------------------------------------------------
# data_transform
# ---------------------------------------------------------------------------


def build_data_transform_node(
    ctx: NodeContext,
    cel: CelEngine,
) -> Callable[..., Any]:
    """Build a data_transform node returning partial state."""
    args: DataTransformArgs = ctx.node_def.args  # type: ignore[assignment]

    async def data_transform_fn(
        _ctx: NodeContext, state: dict[str, Any]
    ) -> dict[str, Any]:
        hooks = _ctx.hook_manager

        try:
            # node_enter
            if hooks:
                from sherma.hooks.types import NodeEnterContext

                await hooks.run_hook(
                    "node_enter",
                    NodeEnterContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        state=state,
                    ),
                )

            raw = cel.evaluate(args.expression, state)
            result: dict[str, Any] = raw if isinstance(raw, dict) else {"result": raw}

            # node_exit
            if hooks:
                from sherma.hooks.types import NodeExitContext

                exit_ctx = await hooks.run_hook(
                    "node_exit",
                    NodeExitContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        result=result,
                        state=state,
                    ),
                )
                result = exit_ctx.result

            return result
        except Exception as exc:
            if isinstance(exc, GraphBubbleUp):
                raise
            return await _run_node_error_hook(hooks, _ctx, state, exc)

    return partial(data_transform_fn, ctx)


# ---------------------------------------------------------------------------
# set_state
# ---------------------------------------------------------------------------


def build_set_state_node(
    ctx: NodeContext,
    cel: CelEngine,
) -> Callable[..., Any]:
    """Build a set_state node that evaluates CEL expressions for each key."""
    args: SetStateArgs = ctx.node_def.args  # type: ignore[assignment]

    async def set_state_fn(_ctx: NodeContext, state: dict[str, Any]) -> dict[str, Any]:
        hooks = _ctx.hook_manager

        try:
            # node_enter
            if hooks:
                from sherma.hooks.types import NodeEnterContext

                await hooks.run_hook(
                    "node_enter",
                    NodeEnterContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        state=state,
                    ),
                )

            result: dict[str, Any] = {}
            for key, expr in args.values.items():
                result[key] = cel.evaluate(expr, state)

            # node_exit
            if hooks:
                from sherma.hooks.types import NodeExitContext

                exit_ctx = await hooks.run_hook(
                    "node_exit",
                    NodeExitContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        result=result,
                        state=state,
                    ),
                )
                result = exit_ctx.result

            return result
        except Exception as exc:
            if isinstance(exc, GraphBubbleUp):
                raise
            return await _run_node_error_hook(hooks, _ctx, state, exc)

    return partial(set_state_fn, ctx)


# ---------------------------------------------------------------------------
# interrupt
# ---------------------------------------------------------------------------


def build_interrupt_node(
    ctx: NodeContext,
    cel: CelEngine,
) -> Callable[..., Any]:
    """Build an interrupt node that pauses graph execution for human input.

    The interrupt value is always the result of evaluating the
    ``args.value`` CEL expression against the current state.
    """
    args: InterruptArgs = ctx.node_def.args  # type: ignore[assignment]

    async def interrupt_fn(_ctx: NodeContext, state: dict[str, Any]) -> dict[str, Any]:
        hooks = _ctx.hook_manager

        try:
            # node_enter
            if hooks:
                from sherma.hooks.types import NodeEnterContext

                await hooks.run_hook(
                    "node_enter",
                    NodeEnterContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        state=state,
                    ),
                )

            value: Any = cel.evaluate(args.value, state)

            # before_interrupt
            if hooks:
                from sherma.hooks.types import BeforeInterruptContext

                before_ctx = await hooks.run_hook(
                    "before_interrupt",
                    BeforeInterruptContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        value=value,
                        state=state,
                    ),
                )
                value = before_ctx.value

            response = interrupt(value)

            # after_interrupt
            if hooks:
                from sherma.hooks.types import AfterInterruptContext

                after_ctx = await hooks.run_hook(
                    "after_interrupt",
                    AfterInterruptContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        value=value,
                        response=response,
                        state=state,
                    ),
                )
                response = after_ctx.response

            result: dict[str, Any] = {"messages": [HumanMessage(content=str(response))]}

            # node_exit
            if hooks:
                from sherma.hooks.types import NodeExitContext

                exit_ctx = await hooks.run_hook(
                    "node_exit",
                    NodeExitContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        result=result,
                        state=state,
                    ),
                )
                result = exit_ctx.result

            return result
        except Exception as exc:
            if isinstance(exc, GraphBubbleUp):
                raise
            return await _run_node_error_hook(hooks, _ctx, state, exc)

    return partial(interrupt_fn, ctx)


# ---------------------------------------------------------------------------
# load_skills
# ---------------------------------------------------------------------------


def build_load_skills_node(
    ctx: NodeContext,
    cel: CelEngine,
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
) -> Callable[..., Any]:
    """Build a load_skills node that programmatically loads skills.

    Evaluates the ``skill_ids`` CEL expression to get a list of
    ``{id, version}`` objects, loads each skill's SKILL.md, registers
    their tools, and synthesizes ``AIMessage(tool_calls)`` +
    ``ToolMessage`` pairs into ``state.messages``.
    """
    args: LoadSkillsArgs = ctx.node_def.args  # type: ignore[assignment]

    async def load_skills_fn(
        _ctx: NodeContext, state: dict[str, Any]
    ) -> dict[str, Any]:
        hooks = _ctx.hook_manager

        try:
            # node_enter
            if hooks:
                from sherma.hooks.types import NodeEnterContext

                await hooks.run_hook(
                    "node_enter",
                    NodeEnterContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        state=state,
                    ),
                )

            # Evaluate CEL to get list of {id, version} dicts
            raw = cel.evaluate(args.skill_ids, state)
            if not isinstance(raw, list):
                raise ValueError(
                    f"skill_ids CEL expression must evaluate to a list, "
                    f"got {type(raw).__name__}"
                )

            from sherma.langgraph.skill_tools import load_and_register_skill

            internal = _get_internal(state)
            current_ids: list[str] = list(internal.get("loaded_tools_from_skills", []))

            # Build tool_calls for the AIMessage and collect ToolMessages
            tool_calls: list[dict[str, Any]] = []
            tool_messages: list[ToolMessage] = []

            for item in raw:
                if not isinstance(item, dict) or "id" not in item:
                    logger.warning(
                        "[%s] Skipping invalid skill entry: %r",
                        _ctx.node_def.name,
                        item,
                    )
                    continue

                skill_id = item["id"]
                version = item.get("version", "*")
                call_id = f"load_skill_{uuid.uuid4().hex[:8]}"

                try:
                    content, tool_ids = await load_and_register_skill(
                        skill_id,
                        version,
                        skill_registry,
                        tool_registry,
                        hooks,
                    )
                except Exception:
                    logger.warning(
                        "[%s] Failed to load skill '%s'",
                        _ctx.node_def.name,
                        skill_id,
                        exc_info=True,
                    )
                    continue

                tool_calls.append(
                    {
                        "id": call_id,
                        "name": "load_skill_md",
                        "args": {"skill_id": skill_id, "version": version},
                    }
                )
                tool_messages.append(ToolMessage(content=content, tool_call_id=call_id))

                for tid in tool_ids:
                    if tid not in current_ids:
                        current_ids.append(tid)

            # Build result messages
            messages: list[Any] = []
            if tool_calls:
                messages.append(AIMessage(content="", tool_calls=tool_calls))
                messages.extend(tool_messages)

            internal["loaded_tools_from_skills"] = current_ids
            result: dict[str, Any] = {"messages": messages}
            _set_internal(result, internal)

            logger.info(
                "[%s] Loaded %d skill(s), %d tool(s) registered",
                _ctx.node_def.name,
                len(tool_calls),
                len(current_ids),
            )

            # node_exit
            if hooks:
                from sherma.hooks.types import NodeExitContext

                exit_ctx = await hooks.run_hook(
                    "node_exit",
                    NodeExitContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        result=result,
                        state=state,
                    ),
                )
                result = exit_ctx.result

            return result
        except Exception as exc:
            if isinstance(exc, GraphBubbleUp):
                raise
            return await _run_node_error_hook(hooks, _ctx, state, exc)

    return partial(load_skills_fn, ctx)


# ---------------------------------------------------------------------------
# custom
# ---------------------------------------------------------------------------


def build_custom_node(
    ctx: NodeContext,
) -> Callable[..., Any]:
    """Build a custom node whose logic is defined entirely by hooks.

    The lifecycle is ``node_enter`` → ``node_execute`` → ``node_exit``.
    The ``node_execute`` hook is unique to custom nodes and is where the
    user-supplied Python logic runs.
    """

    async def custom_fn(_ctx: NodeContext, state: dict[str, Any]) -> dict[str, Any]:
        hooks = _ctx.hook_manager

        try:
            # node_enter
            if hooks:
                from sherma.hooks.types import NodeEnterContext

                await hooks.run_hook(
                    "node_enter",
                    NodeEnterContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        state=state,
                    ),
                )

            # node_execute (custom nodes only)
            result: dict[str, Any] = {}
            if hooks:
                from sherma.hooks.types import NodeExecuteContext

                exec_ctx = await hooks.run_hook(
                    "node_execute",
                    NodeExecuteContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        state=state,
                    ),
                )
                result = exec_ctx.result

            # node_exit
            if hooks:
                from sherma.hooks.types import NodeExitContext

                exit_ctx = await hooks.run_hook(
                    "node_exit",
                    NodeExitContext(
                        node_context=_ctx,
                        node_name=_ctx.node_def.name,
                        node_type=_ctx.node_def.type,
                        result=result,
                        state=state,
                    ),
                )
                result = exit_ctx.result

            return result
        except Exception as exc:
            if isinstance(exc, GraphBubbleUp):
                raise
            return await _run_node_error_hook(hooks, _ctx, state, exc)

    return partial(custom_fn, ctx)


# ---------------------------------------------------------------------------
# Tool resolution helpers
# ---------------------------------------------------------------------------


async def resolve_tools_for_node_async(
    tool_refs: list[Any],
    tool_registry: Any,
) -> list[BaseTool]:
    """Resolve tool refs to LangChain BaseTool instances (async)."""
    resolved: list[BaseTool] = []
    for ref in tool_refs:
        version_spec = f"=={ref.version}" if ref.version and ref.version != "*" else "*"
        tool_entity = await tool_registry.get(ref.id, version_spec)
        resolved.append(to_langgraph_tool(tool_entity))
    return resolved
