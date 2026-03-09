"""Tests for declarative node builders."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from sherma.langgraph.declarative.cel_engine import CelEngine
from sherma.langgraph.declarative.nodes import (
    build_call_llm_node,
    build_data_transform_node,
    build_set_state_node,
    build_tool_node,
)
from sherma.langgraph.declarative.schema import (
    CallLLMArgs,
    DataTransformArgs,
    NodeDef,
    RegistryRef,
    SetStateArgs,
    ToolNodeArgs,
)


@pytest.mark.asyncio
async def test_build_call_llm_node():
    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt='"You are helpful"',
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))
    cel = CelEngine()

    fn = build_call_llm_node(node_def, chat_model, cel)
    result = await fn({"messages": []})

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Hello!"
    chat_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_build_call_llm_node_with_tools():
    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt='"You are helpful"',
            tools=[RegistryRef(id="my-tool")],
        ),
    )
    bound_model = AsyncMock()
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="Using tool"))
    chat_model = MagicMock()
    chat_model.bind_tools = MagicMock(return_value=bound_model)

    mock_tool = MagicMock()
    cel = CelEngine()

    fn = build_call_llm_node(node_def, chat_model, cel, tools=[mock_tool])
    result = await fn({"messages": []})

    chat_model.bind_tools.assert_called_once_with([mock_tool])
    assert result["messages"][0].content == "Using tool"


def test_build_tool_node():
    node_def = NodeDef(
        name="tools",
        type="tool_node",
        args=ToolNodeArgs(tools=[RegistryRef(id="my-tool")]),
    )
    # Need a real BaseTool-like object for ToolNode
    from langchain_core.tools import StructuredTool

    def dummy_fn(x: str) -> str:
        """Dummy tool."""
        return x

    real_tool = StructuredTool.from_function(func=dummy_fn, name="my-tool")
    tn = build_tool_node(node_def, [real_tool])
    assert tn is not None


@pytest.mark.asyncio
async def test_build_data_transform_node():
    node_def = NodeDef(
        name="transform",
        type="data_transform",
        args=DataTransformArgs(expression='{"result": "done"}'),
    )
    cel = CelEngine()
    fn = build_data_transform_node(node_def, cel)
    result = await fn({})
    assert result == {"result": "done"}


@pytest.mark.asyncio
async def test_build_data_transform_non_dict():
    node_def = NodeDef(
        name="transform",
        type="data_transform",
        args=DataTransformArgs(expression='"just a string"'),
    )
    cel = CelEngine()
    fn = build_data_transform_node(node_def, cel)
    result = await fn({})
    assert result == {"result": "just a string"}


@pytest.mark.asyncio
async def test_build_set_state_node():
    node_def = NodeDef(
        name="setter",
        type="set_state",
        args=SetStateArgs(values={"count": "x + 1", "label": '"done"'}),
    )
    cel = CelEngine()
    fn = build_set_state_node(node_def, cel)
    result = await fn({"x": 5})
    assert result == {"count": 6, "label": "done"}
