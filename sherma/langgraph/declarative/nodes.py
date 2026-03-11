"""Node factory functions for declarative agents."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from sherma.hooks.manager import HookManager
from sherma.langgraph.declarative.cel_engine import CelEngine
from sherma.langgraph.declarative.schema import RegistryRef
from sherma.langgraph.tools import to_langgraph_tool
from sherma.logging import get_logger

_BARE_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")

INTERNAL_STATE_KEY = "__sherma__"
"""Top-level state key for all sherma-managed internal data."""

logger = get_logger(__name__)


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
        NodeDef,
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
            current_tools = await _resolve_skill_tools_from_state(state, tool_registry)
        elif args.use_tools_from_registry and tool_registry is not None:
            current_tools = await _resolve_all_registry_tools(tool_registry)
        elif args.use_sub_agents_as_tools and tool_registry is not None:
            sub_agent_tool_ids: list[str] = _ctx.extra.get("sub_agent_tool_ids", [])
            if sub_agent_tool_ids:
                refs = [RegistryRef(id=tid) for tid in sub_agent_tool_ids]
                current_tools = await resolve_tools_for_node_async(refs, tool_registry)

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

        prompt_text = cel.evaluate(args.prompt, state)
        messages = state.get("messages", [])

        # before_llm_call
        if hooks:
            from sherma.hooks.types import BeforeLLMCallContext

            before_ctx = await hooks.run_hook(
                "before_llm_call",
                BeforeLLMCallContext(
                    node_context=_ctx,
                    node_name=_ctx.node_def.name,
                    messages=messages,
                    system_prompt=str(prompt_text),
                    tools=current_tools,
                    state=state,
                ),
            )
            messages = before_ctx.messages
            prompt_text = before_ctx.system_prompt
            current_tools = before_ctx.tools

        model: Any = chat_model
        if current_tools:
            model = model.bind_tools(current_tools)

        system_msg = SystemMessage(content=str(prompt_text))
        logger.info(
            "[%s] Invoking LLM (%d tools) with %d messages, system prompt: %.100s...",
            _ctx.node_def.name,
            len(current_tools),
            len(messages),
            str(prompt_text),
        )
        response = await model.ainvoke([system_msg, *messages])

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
            current_ids: list[str] = list(internal.get("loaded_tools_from_skills", []))
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
                text_parts = [p.root.text for p in last.parts if p.root.kind == "text"]
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

    return partial(set_state_fn, ctx)


# ---------------------------------------------------------------------------
# interrupt
# ---------------------------------------------------------------------------


def _find_last_ai_message(state: dict[str, Any]) -> AIMessage | None:
    """Return the last AIMessage with content from state, or ``None``."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and getattr(msg, "content", ""):
            return msg
    return None


def build_interrupt_node(
    ctx: NodeContext,
    cel: CelEngine,
) -> Callable[..., Any]:
    """Build an interrupt node that pauses graph execution for human input.

    The interrupt value is the last ``AIMessage`` from state when
    available.  Falls back to evaluating the ``args.value`` CEL
    expression when no AIMessage is present.
    """
    args: InterruptArgs = ctx.node_def.args  # type: ignore[assignment]

    async def interrupt_fn(_ctx: NodeContext, state: dict[str, Any]) -> dict[str, Any]:
        hooks = _ctx.hook_manager

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

        value: Any = _find_last_ai_message(state)
        if value is None and args and args.value is not None:
            value = cel.evaluate(args.value, state)

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

    return partial(interrupt_fn, ctx)


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
