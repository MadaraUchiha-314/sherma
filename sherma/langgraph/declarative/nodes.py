"""Node factory functions for declarative agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from sherma.langgraph.declarative.cel_engine import CelEngine
from sherma.langgraph.tools import to_langgraph_tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from sherma.langgraph.declarative.schema import (
        CallAgentArgs,
        CallLLMArgs,
        DataTransformArgs,
        NodeDef,
        SetStateArgs,
    )


def build_call_llm_node(
    node_def: NodeDef,
    chat_model: BaseChatModel,
    cel: CelEngine,
    tools: list[BaseTool] | None = None,
) -> Callable[..., Any]:
    """Build a call_llm node function."""
    args: CallLLMArgs = node_def.args  # type: ignore[assignment]
    model = chat_model
    if tools:
        model = model.bind_tools(tools)  # type: ignore[assignment]

    async def call_llm_fn(state: dict[str, Any]) -> dict[str, Any]:
        prompt_text = cel.evaluate(args.prompt, state)
        system_msg = SystemMessage(content=str(prompt_text))
        messages = state.get("messages", [])
        response = await model.ainvoke([system_msg, *messages])
        return {"messages": [response]}

    return call_llm_fn


def build_tool_node(
    node_def: NodeDef,
    tools: list[BaseTool],
) -> ToolNode:
    """Build a tool_node that executes tool calls."""
    return ToolNode(tools)


def build_call_agent_node(
    node_def: NodeDef,
    agent: Any,
    cel: CelEngine,
) -> Callable[..., Any]:
    """Build a call_agent node function."""
    args: CallAgentArgs = node_def.args  # type: ignore[assignment]

    async def call_agent_fn(state: dict[str, Any]) -> dict[str, Any]:
        input_val = cel.evaluate(args.input, state)
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
        if results:
            last = results[-1]
            if isinstance(last, A2AMessage):
                text_parts = [p.root.text for p in last.parts if p.root.kind == "text"]
                content = " ".join(text_parts)
                return {"messages": [AIMessage(content=content)]}
        return {}

    return call_agent_fn


def build_data_transform_node(
    node_def: NodeDef,
    cel: CelEngine,
) -> Callable[..., Any]:
    """Build a data_transform node returning partial state."""
    args: DataTransformArgs = node_def.args  # type: ignore[assignment]

    async def data_transform_fn(state: dict[str, Any]) -> dict[str, Any]:
        result = cel.evaluate(args.expression, state)
        if isinstance(result, dict):
            return result
        return {"result": result}

    return data_transform_fn


def build_set_state_node(
    node_def: NodeDef,
    cel: CelEngine,
) -> Callable[..., Any]:
    """Build a set_state node that evaluates CEL expressions for each key."""
    args: SetStateArgs = node_def.args  # type: ignore[assignment]

    async def set_state_fn(state: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, expr in args.values.items():
            result[key] = cel.evaluate(expr, state)
        return result

    return set_state_fn


def resolve_tools_for_node(
    tool_refs: list[Any],
    tool_registry: Any,
) -> list[BaseTool]:
    """Resolve tool registry references to LangChain BaseTool instances."""
    resolved: list[BaseTool] = []
    for ref in tool_refs:
        # Tools must already be in the registry as sherma Tool instances
        import asyncio

        version_spec = f"=={ref.version}" if ref.version else "*"
        tool_entity = asyncio.get_event_loop().run_until_complete(
            tool_registry.get(ref.id, version_spec)
        )
        resolved.append(to_langgraph_tool(tool_entity))
    return resolved


async def resolve_tools_for_node_async(
    tool_refs: list[Any],
    tool_registry: Any,
) -> list[BaseTool]:
    """Resolve tool refs to LangChain BaseTool instances (async)."""
    resolved: list[BaseTool] = []
    for ref in tool_refs:
        version_spec = f"=={ref.version}" if ref.version else "*"
        tool_entity = await tool_registry.get(ref.id, version_spec)
        resolved.append(to_langgraph_tool(tool_entity))
    return resolved
