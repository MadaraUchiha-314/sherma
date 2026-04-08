"""Tests for declarative node builders."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.manager import HookManager
from sherma.hooks.types import (
    AfterLLMCallContext,
    BeforeLLMCallContext,
    NodeEnterContext,
    NodeExitContext,
)
from sherma.langgraph.declarative.cel_engine import CelEngine
from sherma.langgraph.declarative.nodes import (
    INTERNAL_STATE_KEY,
    NodeContext,
    _resolve_all_registry_tools,
    _resolve_skill_tools_from_state,
    build_call_llm_node,
    build_data_transform_node,
    build_interrupt_node,
    build_load_skills_node,
    build_set_state_node,
    build_tool_node,
)
from sherma.langgraph.declarative.schema import (
    CallLLMArgs,
    DataTransformArgs,
    DeclarativeConfig,
    InterruptArgs,
    LoadSkillsArgs,
    NodeDef,
    PromptMessageDef,
    RegistryRef,
    SetStateArgs,
    ToolNodeArgs,
)


def _make_ctx(
    node_def: NodeDef, hook_manager: HookManager | None = None
) -> NodeContext:
    """Create a minimal NodeContext for testing."""
    config = DeclarativeConfig(manifest_version=1)
    return NodeContext(config=config, node_def=node_def, hook_manager=hook_manager)


@pytest.mark.asyncio
async def test_build_call_llm_node():
    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            state_updates={"messages": "[llm_response]"},
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))
    cel = CelEngine()

    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    result = await fn({"messages": []})

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0]["content"] == "Hello!"
    chat_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_build_call_llm_node_with_tools():
    """call_llm with explicit tools in args resolves them from registry."""
    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.tool import ToolRegistry

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            tools=[RegistryRef(id="my-tool", version="1.0.0")],
            state_updates={"messages": "[llm_response]"},
        ),
    )
    bound_model = AsyncMock()
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="Using tool"))
    chat_model = MagicMock()
    chat_model.bind_tools = MagicMock(return_value=bound_model)

    registry = ToolRegistry()
    real_tool = StructuredTool.from_function(
        func=lambda x: x, name="my-tool", description="test"
    )
    await registry.add(
        RegistryEntry(
            id="my-tool",
            version="1.0.0",
            instance=Tool(id="my-tool", version="1.0.0", function=real_tool),
        )
    )

    cel = CelEngine()
    fn = build_call_llm_node(
        _make_ctx(node_def), chat_model, cel, tool_registry=registry
    )
    result = await fn({"messages": []})

    chat_model.bind_tools.assert_called_once()
    assert len(chat_model.bind_tools.call_args[0][0]) == 1
    assert result["messages"][0]["content"] == "Using tool"


@pytest.mark.asyncio
async def test_build_tool_node_static():
    """tool_node with explicit tools in args resolves them from registry."""
    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.tool import ToolRegistry

    def dummy_fn(x: str) -> str:
        """Dummy tool."""
        return x

    real_tool = StructuredTool.from_function(func=dummy_fn, name="my-tool")
    registry = ToolRegistry()
    await registry.add(
        RegistryEntry(
            id="my-tool",
            version="1.0.0",
            instance=Tool(id="my-tool", version="1.0.0", function=real_tool),
        )
    )

    node_def = NodeDef(
        name="tools",
        type="tool_node",
        args=ToolNodeArgs(tools=[RegistryRef(id="my-tool", version="1.0.0")]),
    )
    fn = build_tool_node(_make_ctx(node_def), tool_registry=registry)
    assert callable(fn)


@pytest.mark.asyncio
async def test_build_data_transform_node():
    node_def = NodeDef(
        name="transform",
        type="data_transform",
        args=DataTransformArgs(expression='{"result": "done"}'),
    )
    cel = CelEngine()
    fn = build_data_transform_node(_make_ctx(node_def), cel)
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
    fn = build_data_transform_node(_make_ctx(node_def), cel)
    result = await fn({})
    assert result == {"result": "just a string"}


@pytest.mark.asyncio
async def test_build_set_state_node():
    node_def = NodeDef(
        name="setter",
        type="set_state",
        args=SetStateArgs(values={"count": "state.x + 1", "label": '"done"'}),
    )
    cel = CelEngine()
    fn = build_set_state_node(_make_ctx(node_def), cel)
    result = await fn({"x": 5})
    assert result == {"count": 6, "label": "done"}


# --- NodeContext tests ---


def test_node_context_has_config():
    """NodeContext exposes the config and node_def."""
    node_def = NodeDef(
        name="test",
        type="set_state",
        args=SetStateArgs(values={"x": '"hi"'}),
    )
    config = DeclarativeConfig(manifest_version=1)
    ctx = NodeContext(config=config, node_def=node_def)
    assert ctx.config is config
    assert ctx.node_def is node_def
    assert ctx.extra == {}


# --- Dynamic tool resolution tests ---


@pytest.mark.asyncio
async def test_resolve_skill_tools_from_state():
    """_resolve_skill_tools_from_state returns tools for IDs in state."""
    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.tool import ToolRegistry

    registry = ToolRegistry()
    real_tool = StructuredTool.from_function(
        func=lambda x: x, name="my-tool", description="test"
    )
    await registry.add(
        RegistryEntry(
            id="my-tool",
            version="1.0.0",
            instance=Tool(id="my-tool", version="1.0.0", function=real_tool),
        )
    )

    state: dict[str, object] = {
        INTERNAL_STATE_KEY: {"loaded_tools_from_skills": ["my-tool"]}
    }
    resolved = await _resolve_skill_tools_from_state(state, registry)
    assert len(resolved) == 1
    assert resolved[0].name == "my-tool"


@pytest.mark.asyncio
async def test_resolve_skill_tools_from_state_empty():
    """_resolve_skill_tools_from_state returns empty list when no IDs."""
    from sherma.registry.tool import ToolRegistry

    registry = ToolRegistry()
    resolved = await _resolve_skill_tools_from_state({}, registry)
    assert resolved == []


@pytest.mark.asyncio
async def test_resolve_skill_tools_from_state_missing_tool():
    """_resolve_skill_tools_from_state skips unresolvable tool IDs."""
    from sherma.registry.tool import ToolRegistry

    registry = ToolRegistry()
    state: dict[str, object] = {
        INTERNAL_STATE_KEY: {"loaded_tools_from_skills": ["nonexistent"]}
    }
    resolved = await _resolve_skill_tools_from_state(state, registry)
    assert resolved == []


@pytest.mark.asyncio
async def test_resolve_all_registry_tools():
    """_resolve_all_registry_tools returns all tools in registry."""
    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.tool import ToolRegistry

    registry = ToolRegistry()
    tool1 = StructuredTool.from_function(
        func=lambda x: x, name="tool-a", description="a"
    )
    tool2 = StructuredTool.from_function(
        func=lambda x: x, name="tool-b", description="b"
    )
    await registry.add(
        RegistryEntry(
            id="tool-a",
            version="1.0.0",
            instance=Tool(id="tool-a", version="1.0.0", function=tool1),
        )
    )
    await registry.add(
        RegistryEntry(
            id="tool-b",
            version="1.0.0",
            instance=Tool(id="tool-b", version="1.0.0", function=tool2),
        )
    )

    resolved = await _resolve_all_registry_tools(registry)
    assert len(resolved) == 2
    names = {t.name for t in resolved}
    assert names == {"tool-a", "tool-b"}


@pytest.mark.asyncio
async def test_build_call_llm_node_use_tools_from_loaded_skills():
    """use_tools_from_loaded_skills reads _skill_tool_ids and binds tools."""
    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.tool import ToolRegistry

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            use_tools_from_loaded_skills=True,
            state_updates={"messages": "[llm_response]"},
        ),
    )

    bound_model = AsyncMock()
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="Using skill tool"))
    chat_model = MagicMock()
    chat_model.bind_tools = MagicMock(return_value=bound_model)

    registry = ToolRegistry()
    real_tool = StructuredTool.from_function(
        func=lambda x: x, name="skill-tool", description="A skill tool"
    )
    await registry.add(
        RegistryEntry(
            id="skill-tool",
            version="1.0.0",
            instance=Tool(id="skill-tool", version="1.0.0", function=real_tool),
        )
    )

    cel = CelEngine()
    fn = build_call_llm_node(
        _make_ctx(node_def), chat_model, cel, tool_registry=registry
    )

    result = await fn(
        {
            "messages": [],
            INTERNAL_STATE_KEY: {"loaded_tools_from_skills": ["skill-tool"]},
        }
    )

    chat_model.bind_tools.assert_called_once()
    bound_tools = chat_model.bind_tools.call_args[0][0]
    assert len(bound_tools) == 1
    assert bound_tools[0].name == "skill-tool"
    assert result["messages"][0]["content"] == "Using skill tool"


@pytest.mark.asyncio
async def test_build_call_llm_node_loaded_skills_plus_explicit_tools():
    """use_tools_from_loaded_skills + explicit tools merges both sets with dedup."""
    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.tool import ToolRegistry

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            use_tools_from_loaded_skills=True,
            tools=[RegistryRef(id="explicit-tool", version="1.0.0")],
            state_updates={"messages": "[llm_response]"},
        ),
    )

    bound_model = AsyncMock()
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="merged"))
    chat_model = MagicMock()
    chat_model.bind_tools = MagicMock(return_value=bound_model)

    registry = ToolRegistry()
    skill_tool = StructuredTool.from_function(
        func=lambda x: x, name="skill-tool", description="A skill tool"
    )
    explicit_tool = StructuredTool.from_function(
        func=lambda x: x, name="explicit-tool", description="An explicit tool"
    )
    # Also add a duplicate under both paths to verify dedup
    await registry.add(
        RegistryEntry(
            id="skill-tool",
            version="1.0.0",
            instance=Tool(id="skill-tool", version="1.0.0", function=skill_tool),
        )
    )
    await registry.add(
        RegistryEntry(
            id="explicit-tool",
            version="1.0.0",
            instance=Tool(id="explicit-tool", version="1.0.0", function=explicit_tool),
        )
    )

    cel = CelEngine()
    fn = build_call_llm_node(
        _make_ctx(node_def), chat_model, cel, tool_registry=registry
    )

    result = await fn(
        {
            "messages": [],
            INTERNAL_STATE_KEY: {"loaded_tools_from_skills": ["skill-tool"]},
        }
    )

    chat_model.bind_tools.assert_called_once()
    bound_tools = chat_model.bind_tools.call_args[0][0]
    names = {t.name for t in bound_tools}
    assert names == {"skill-tool", "explicit-tool"}
    assert len(bound_tools) == 2  # no duplicates
    assert result["messages"][0]["content"] == "merged"


@pytest.mark.asyncio
async def test_build_call_llm_node_use_tools_from_loaded_skills_empty():
    """use_tools_from_loaded_skills with empty _skill_tool_ids doesn't bind."""
    from sherma.registry.tool import ToolRegistry

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            use_tools_from_loaded_skills=True,
            state_updates={"messages": "[llm_response]"},
        ),
    )

    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="No tools"))

    registry = ToolRegistry()
    cel = CelEngine()
    fn = build_call_llm_node(
        _make_ctx(node_def), chat_model, cel, tool_registry=registry
    )

    result = await fn(
        {
            "messages": [],
            INTERNAL_STATE_KEY: {"loaded_tools_from_skills": []},
        }
    )

    assert result["messages"][0]["content"] == "No tools"
    chat_model.bind_tools.assert_not_called()


@pytest.mark.asyncio
async def test_build_call_llm_node_use_tools_from_registry():
    """use_tools_from_registry binds ALL tools from the registry."""
    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.tool import ToolRegistry

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            use_tools_from_registry=True,
            state_updates={"messages": "[llm_response]"},
        ),
    )

    bound_model = AsyncMock()
    bound_model.ainvoke = AsyncMock(
        return_value=AIMessage(content="Using registry tools")
    )
    chat_model = MagicMock()
    chat_model.bind_tools = MagicMock(return_value=bound_model)

    registry = ToolRegistry()
    for name in ["tool-a", "tool-b"]:
        real_tool = StructuredTool.from_function(
            func=lambda x: x, name=name, description=name
        )
        await registry.add(
            RegistryEntry(
                id=name,
                version="1.0.0",
                instance=Tool(id=name, version="1.0.0", function=real_tool),
            )
        )

    cel = CelEngine()
    fn = build_call_llm_node(
        _make_ctx(node_def), chat_model, cel, tool_registry=registry
    )

    result = await fn({"messages": []})

    chat_model.bind_tools.assert_called_once()
    bound_tools = chat_model.bind_tools.call_args[0][0]
    assert len(bound_tools) == 2
    assert result["messages"][0]["content"] == "Using registry tools"


@pytest.mark.asyncio
async def test_build_tool_node_registry():
    """tool_node with registry resolves tools at invocation time."""
    from unittest.mock import patch

    from langchain_core.messages import ToolMessage
    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.tool import ToolRegistry

    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    real_tool = StructuredTool.from_function(func=greet, name="greet")
    registry = ToolRegistry()
    await registry.add(
        RegistryEntry(
            id="greet",
            version="1.0.0",
            instance=Tool(id="greet", version="1.0.0", function=real_tool),
        )
    )

    node_def = NodeDef(
        name="tools",
        type="tool_node",
        args=ToolNodeArgs(),
    )

    tool_msg = ToolMessage(content="Hello, World!", tool_call_id="call_1")
    mock_result: dict[str, object] = {"messages": [tool_msg]}
    mock_tool_node = MagicMock()
    mock_tool_node.ainvoke = AsyncMock(return_value=mock_result)

    fn = build_tool_node(_make_ctx(node_def), tool_registry=registry)

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "greet", "args": {"name": "World"}}],
    )

    with patch(
        "sherma.langgraph.declarative.nodes.ToolNode",
        return_value=mock_tool_node,
    ):
        result = await fn({"messages": [ai_msg]})

    assert "messages" in result
    assert len(result["messages"]) >= 1


@pytest.mark.asyncio
async def test_build_tool_node_registry_empty():
    """tool_node with empty registry returns empty messages."""
    from sherma.registry.tool import ToolRegistry

    registry = ToolRegistry()

    node_def = NodeDef(
        name="tools",
        type="tool_node",
        args=ToolNodeArgs(),
    )

    fn = build_tool_node(_make_ctx(node_def), tool_registry=registry)
    result = await fn({"messages": []})

    assert result == {"messages": []}


@pytest.mark.asyncio
async def test_build_tool_node_with_skill_card_tracking():
    """tool_node with skill_registry tracks _skill_tool_ids."""
    from unittest.mock import patch

    from langchain_core.messages import ToolMessage
    from langchain_core.tools import StructuredTool

    from sherma.entities.skill import Skill, SkillFrontMatter
    from sherma.entities.skill_card import LocalToolDef, SkillCard
    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.skill import SkillRegistry
    from sherma.registry.tool import ToolRegistry

    card = SkillCard(
        id="weather",
        version="1.0.0",
        name="Weather",
        description="Weather skill",
        base_uri=".",
        local_tools={
            "get_weather": LocalToolDef(
                id="get_weather",
                version="1.0.0",
                import_path="examples.tools.get_weather",
            )
        },
    )
    skill = Skill(
        id="weather",
        version="1.0.0",
        front_matter=SkillFrontMatter(name="Weather", description="Weather skill"),
        skill_card=card,
    )
    skill_registry = SkillRegistry()
    await skill_registry.add(
        RegistryEntry(id="weather", version="1.0.0", instance=skill)
    )

    def load_skill_md(skill_id: str, version: str = "*") -> str:
        """Load a skill."""
        return "loaded"

    real_tool = StructuredTool.from_function(func=load_skill_md, name="load_skill_md")
    tool_registry = ToolRegistry()
    await tool_registry.add(
        RegistryEntry(
            id="load_skill_md",
            version="1.0.0",
            instance=Tool(id="load_skill_md", version="1.0.0", function=real_tool),
        )
    )

    node_def = NodeDef(
        name="tools",
        type="tool_node",
        args=ToolNodeArgs(),
    )

    tool_msg = ToolMessage(content="loaded", tool_call_id="call_1")
    mock_result: dict[str, object] = {"messages": [tool_msg]}
    mock_tool_node = MagicMock()
    mock_tool_node.ainvoke = AsyncMock(return_value=mock_result)

    fn = build_tool_node(
        _make_ctx(node_def),
        tool_registry=tool_registry,
        skill_registry=skill_registry,
    )

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "load_skill_md",
                "args": {"skill_id": "weather", "version": "1.0.0"},
            }
        ],
    )
    state: dict[str, object] = {
        "messages": [ai_msg],
        INTERNAL_STATE_KEY: {"loaded_tools_from_skills": []},
    }

    with patch(
        "sherma.langgraph.declarative.nodes.ToolNode",
        return_value=mock_tool_node,
    ):
        result = await fn(state)

    internal = result[INTERNAL_STATE_KEY]
    assert "get_weather" in internal["loaded_tools_from_skills"]


@pytest.mark.asyncio
async def test_node_receives_config_via_partial():
    """Verify that node functions receive the config through partial injection."""
    from functools import partial as stdlib_partial

    node_def = NodeDef(
        name="checker",
        type="set_state",
        args=SetStateArgs(values={"x": '"hi"'}),
    )
    config = DeclarativeConfig(manifest_version=1)
    ctx = NodeContext(config=config, node_def=node_def)
    cel = CelEngine()

    fn = build_set_state_node(ctx, cel)

    # The returned function should be a functools.partial with ctx bound
    assert isinstance(fn, stdlib_partial)
    assert fn.args[0] is ctx
    assert fn.args[0].config is config


# --- Interrupt node tests ---


@pytest.mark.asyncio
async def test_build_interrupt_node_evaluates_cel_value():
    """interrupt node always evaluates the CEL args.value expression."""
    from unittest.mock import patch

    from langchain_core.messages import HumanMessage

    node_def = NodeDef(
        name="ask",
        type="interrupt",
        args=InterruptArgs(value='"What is your name?"'),
    )
    cel = CelEngine()
    fn = build_interrupt_node(_make_ctx(node_def), cel)

    with patch(
        "sherma.langgraph.declarative.nodes.interrupt",
        return_value="Alice",
    ) as mock_interrupt:
        result = await fn({"messages": []})

    mock_interrupt.assert_called_once_with("What is your name?")
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "Alice"


@pytest.mark.asyncio
async def test_build_interrupt_node_cel_uses_state():
    """interrupt node CEL expression can reference state values."""
    from unittest.mock import patch

    from langchain_core.messages import AIMessage, HumanMessage

    node_def = NodeDef(
        name="ask_confirm",
        type="interrupt",
        args=InterruptArgs(value="state.messages[size(state.messages) - 1].content"),
    )
    cel = CelEngine()
    fn = build_interrupt_node(_make_ctx(node_def), cel)

    ai_msg = AIMessage(content="Please confirm")
    with patch(
        "sherma.langgraph.declarative.nodes.interrupt",
        return_value="yes",
    ) as mock_interrupt:
        result = await fn({"messages": [ai_msg]})

    mock_interrupt.assert_called_once_with("Please confirm")
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "yes"


@pytest.mark.asyncio
async def test_build_interrupt_node_cel_structured_metadata():
    """interrupt node can pass structured metadata via CEL map literal."""
    from unittest.mock import patch

    from langchain_core.messages import HumanMessage

    cel_expr = '{"type": "approval", "actions": ["approve", "reject"]}'
    node_def = NodeDef(
        name="ask_approval",
        type="interrupt",
        args=InterruptArgs(value=cel_expr),
    )
    cel = CelEngine()
    fn = build_interrupt_node(_make_ctx(node_def), cel)

    with patch(
        "sherma.langgraph.declarative.nodes.interrupt",
        return_value="approve",
    ) as mock_interrupt:
        result = await fn({"messages": []})

    expected = {"type": "approval", "actions": ["approve", "reject"]}
    mock_interrupt.assert_called_once_with(expected)
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "approve"


# --- Hook integration tests ---


@pytest.mark.asyncio
async def test_call_llm_fires_hooks():
    """call_llm fires node_enter, before_llm_call, after_llm_call, node_exit."""
    events: list[str] = []

    class TrackingHook(BaseHookExecutor):
        async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
            events.append("node_enter")
            return None

        async def before_llm_call(
            self, ctx: BeforeLLMCallContext
        ) -> BeforeLLMCallContext | None:
            events.append("before_llm_call")
            return None

        async def after_llm_call(
            self, ctx: AfterLLMCallContext
        ) -> AfterLLMCallContext | None:
            events.append("after_llm_call")
            return None

        async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
            events.append("node_exit")
            return None

    hook_manager = HookManager()
    hook_manager.register(TrackingHook())

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            state_updates={"messages": "[llm_response]"},
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))
    cel = CelEngine()

    fn = build_call_llm_node(
        _make_ctx(node_def, hook_manager=hook_manager), chat_model, cel
    )
    await fn({"messages": []})

    assert events == ["node_enter", "before_llm_call", "after_llm_call", "node_exit"]


@pytest.mark.asyncio
async def test_before_llm_call_hook_modifies_prompt():
    """before_llm_call hook can modify the system prompt."""

    class ModifyPrompt(BaseHookExecutor):
        async def before_llm_call(
            self, ctx: BeforeLLMCallContext
        ) -> BeforeLLMCallContext | None:
            ctx.system_prompt = "Modified prompt"
            return ctx

    hook_manager = HookManager()
    hook_manager.register(ModifyPrompt())

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"Original prompt"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            state_updates={"messages": "[llm_response]"},
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
    cel = CelEngine()

    fn = build_call_llm_node(
        _make_ctx(node_def, hook_manager=hook_manager), chat_model, cel
    )
    await fn({"messages": []})

    # Check that the LLM was invoked with the modified prompt
    call_args = chat_model.ainvoke.call_args[0][0]

    assert call_args[0].content == "Modified prompt"


@pytest.mark.asyncio
async def test_node_exit_hook_modifies_result():
    """node_exit hook can modify the result of a data_transform node."""

    class AddExtra(BaseHookExecutor):
        async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
            ctx.result["hook_added"] = True
            return ctx

    hook_manager = HookManager()
    hook_manager.register(AddExtra())

    node_def = NodeDef(
        name="transform",
        type="data_transform",
        args=DataTransformArgs(expression='{"result": "done"}'),
    )
    cel = CelEngine()
    fn = build_data_transform_node(_make_ctx(node_def, hook_manager=hook_manager), cel)
    result = await fn({})
    assert result == {"result": "done", "hook_added": True}


@pytest.mark.asyncio
async def test_build_call_llm_node_with_response_format():
    """call_llm with response_format calls with_structured_output on the model."""
    from sherma.langgraph.declarative.schema import ResponseFormatDef

    node_def = NodeDef(
        name="extract",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"Extract user info"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            response_format=ResponseFormatDef(
                name="UserInfo",
                description="User information",
                **{
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    }
                },
            ),
            state_updates={"messages": "[llm_response]"},
        ),
    )

    structured_model = AsyncMock()
    structured_model.ainvoke = AsyncMock(return_value={"name": "Alice"})

    chat_model = MagicMock()
    chat_model.with_structured_output = MagicMock(return_value=structured_model)

    cel = CelEngine()
    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    result = await fn({"messages": []})

    chat_model.with_structured_output.assert_called_once()
    schema_arg = chat_model.with_structured_output.call_args[0][0]
    assert schema_arg["name"] == "UserInfo"
    assert schema_arg["schema"]["required"] == ["name"]

    # Dict response should be wrapped as AIMessage with JSON content, then
    # CEL round-trips it to a dict with a "content" key containing JSON.
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    import json

    assert json.loads(msg["content"]) == {"name": "Alice"}


@pytest.mark.asyncio
async def test_build_call_llm_node_response_format_aimessage_passthrough():
    """If with_structured_output returns an AIMessage, it passes through."""
    from sherma.langgraph.declarative.schema import ResponseFormatDef

    node_def = NodeDef(
        name="extract",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"Extract"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            response_format=ResponseFormatDef(
                name="Info",
                **{"schema": {"type": "object", "properties": {}}},
            ),
            state_updates={"messages": "[llm_response]"},
        ),
    )

    # Some providers return AIMessage even with structured output
    ai_response = AIMessage(content='{"name": "Bob"}')
    structured_model = AsyncMock()
    structured_model.ainvoke = AsyncMock(return_value=ai_response)

    chat_model = MagicMock()
    chat_model.with_structured_output = MagicMock(return_value=structured_model)

    cel = CelEngine()
    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    result = await fn({"messages": []})

    # AIMessage response passes through CEL, round-tripped to a dict
    assert result["messages"][0]["content"] == '{"name": "Bob"}'


@pytest.mark.asyncio
async def test_no_hooks_when_manager_is_none():
    """When hook_manager is None, nodes work as before without errors."""
    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            state_updates={"messages": "[llm_response]"},
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))
    cel = CelEngine()

    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    result = await fn({"messages": []})
    assert result["messages"][0]["content"] == "Hello!"


# --- Array prompt tests ---


@pytest.mark.asyncio
async def test_array_prompt_with_splice():
    """Array prompt with role=messages splices messages preserving roles."""
    from langchain_core.messages import HumanMessage, SystemMessage

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
                PromptMessageDef(role="human", content='"Summarize the above"'),
            ],
            state_updates={"messages": "[llm_response]"},
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="Summary"))
    cel = CelEngine()

    existing_messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
    ]
    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    await fn({"messages": existing_messages})

    call_args = chat_model.ainvoke.call_args[0][0]
    assert len(call_args) == 4
    assert isinstance(call_args[0], SystemMessage)
    assert call_args[0].content == "You are helpful"
    assert isinstance(call_args[1], HumanMessage)
    assert call_args[1].content == "Hello"
    assert isinstance(call_args[2], AIMessage)
    assert call_args[2].content == "Hi there!"
    assert isinstance(call_args[3], HumanMessage)
    assert call_args[3].content == "Summarize the above"


@pytest.mark.asyncio
async def test_array_prompt_no_auto_messages():
    """Without role=messages, state messages are NOT injected."""
    from langchain_core.messages import HumanMessage, SystemMessage

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"System only"'),
            ],
            state_updates={"messages": "[llm_response]"},
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
    cel = CelEngine()

    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    await fn({"messages": [HumanMessage(content="user msg")]})

    call_args = chat_model.ainvoke.call_args[0][0]
    # Only the system message, no auto-injected user messages
    assert len(call_args) == 1
    assert isinstance(call_args[0], SystemMessage)
    assert call_args[0].content == "System only"


@pytest.mark.asyncio
async def test_array_prompt_mixed_roles():
    """Array prompt with system, human, and ai roles."""
    from langchain_core.messages import HumanMessage, SystemMessage

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"Be concise"'),
                PromptMessageDef(role="human", content='"What is 2+2?"'),
                PromptMessageDef(role="ai", content='"4"'),
                PromptMessageDef(role="human", content='"And 3+3?"'),
            ],
            state_updates={"messages": "[llm_response]"},
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="6"))
    cel = CelEngine()

    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    await fn({"messages": []})

    call_args = chat_model.ainvoke.call_args[0][0]
    assert len(call_args) == 4
    assert isinstance(call_args[0], SystemMessage)
    assert isinstance(call_args[1], HumanMessage)
    assert isinstance(call_args[2], AIMessage)
    assert isinstance(call_args[3], HumanMessage)


# --- use_sub_agents_as_tools tests ---


@pytest.mark.asyncio
async def test_build_call_llm_node_sub_agents_all():
    """use_sub_agents_as_tools='all' resolves all sub-agent tools from context."""
    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.tool import ToolRegistry

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            use_sub_agents_as_tools="all",
            state_updates={"messages": "[llm_response]"},
        ),
    )

    bound_model = AsyncMock()
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="Using sub-agents"))
    chat_model = MagicMock()
    chat_model.bind_tools = MagicMock(return_value=bound_model)

    registry = ToolRegistry()
    for name in ["weather-agent", "search-agent"]:
        real_tool = StructuredTool.from_function(
            func=lambda x: x, name=name, description=name
        )
        await registry.add(
            RegistryEntry(
                id=name,
                version="1.0.0",
                instance=Tool(id=name, version="1.0.0", function=real_tool),
            )
        )

    cel = CelEngine()
    ctx = _make_ctx(node_def)
    ctx.extra["sub_agent_tool_ids"] = ["weather-agent", "search-agent"]
    fn = build_call_llm_node(ctx, chat_model, cel, tool_registry=registry)

    result = await fn({"messages": []})

    chat_model.bind_tools.assert_called_once()
    bound_tools = chat_model.bind_tools.call_args[0][0]
    assert len(bound_tools) == 2
    assert {t.name for t in bound_tools} == {"weather-agent", "search-agent"}
    assert result["messages"][0]["content"] == "Using sub-agents"


@pytest.mark.asyncio
async def test_build_call_llm_node_sub_agents_list():
    """use_sub_agents_as_tools=[RegistryRef] resolves only specific sub-agents."""
    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.tool import ToolRegistry

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            use_sub_agents_as_tools=[
                RegistryRef(id="weather-agent", version="1.0.0"),
            ],
            state_updates={"messages": "[llm_response]"},
        ),
    )

    bound_model = AsyncMock()
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="Subset"))
    chat_model = MagicMock()
    chat_model.bind_tools = MagicMock(return_value=bound_model)

    registry = ToolRegistry()
    for name in ["weather-agent", "search-agent"]:
        real_tool = StructuredTool.from_function(
            func=lambda x: x, name=name, description=name
        )
        await registry.add(
            RegistryEntry(
                id=name,
                version="1.0.0",
                instance=Tool(id=name, version="1.0.0", function=real_tool),
            )
        )

    cel = CelEngine()
    ctx = _make_ctx(node_def)
    ctx.extra["sub_agent_tool_ids"] = ["weather-agent", "search-agent"]
    fn = build_call_llm_node(ctx, chat_model, cel, tool_registry=registry)

    result = await fn({"messages": []})

    chat_model.bind_tools.assert_called_once()
    bound_tools = chat_model.bind_tools.call_args[0][0]
    assert len(bound_tools) == 1
    assert bound_tools[0].name == "weather-agent"
    assert result["messages"][0]["content"] == "Subset"


@pytest.mark.asyncio
async def test_build_call_llm_node_sub_agents_false():
    """use_sub_agents_as_tools=False doesn't bind any sub-agent tools."""
    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"You are helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            use_sub_agents_as_tools=False,
            state_updates={"messages": "[llm_response]"},
        ),
    )

    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="No tools"))

    cel = CelEngine()
    ctx = _make_ctx(node_def)
    ctx.extra["sub_agent_tool_ids"] = ["weather-agent"]
    fn = build_call_llm_node(ctx, chat_model, cel)

    result = await fn({"messages": []})

    assert result["messages"][0]["content"] == "No tools"
    # bind_tools should NOT have been called since it's an AsyncMock (no bind_tools)
    chat_model.ainvoke.assert_called_once()


# --- load_skills node tests ---


def _make_skill_registries():
    """Create skill and tool registries with a test skill."""
    from sherma.entities.skill import Skill, SkillFrontMatter
    from sherma.entities.skill_card import LocalToolDef, SkillCard
    from sherma.registry.skill import SkillRegistry
    from sherma.registry.tool import ToolRegistry

    card = SkillCard(
        id="weather",
        version="1.0.0",
        name="Weather",
        description="Weather skill",
        base_uri=".",
        local_tools={
            "get_weather": LocalToolDef(
                id="get_weather",
                version="1.0.0",
                import_path="examples.tools.get_weather",
            )
        },
    )
    skill = Skill(
        id="weather",
        version="1.0.0",
        front_matter=SkillFrontMatter(name="Weather", description="Weather skill"),
        skill_card=card,
    )
    return skill, card, SkillRegistry(), ToolRegistry()


@pytest.mark.asyncio
async def test_build_load_skills_node_basic():
    """load_skills loads a single skill and produces AIMessage + ToolMessage."""
    from unittest.mock import patch

    from langchain_core.messages import ToolMessage

    from sherma.registry.base import RegistryEntry

    skill, _card, skill_registry, tool_registry = _make_skill_registries()
    await skill_registry.add(
        RegistryEntry(id="weather", version="1.0.0", instance=skill)
    )

    node_def = NodeDef(
        name="load",
        type="load_skills",
        args=LoadSkillsArgs(skill_ids='[{"id": "weather", "version": "1.0.0"}]'),
    )
    cel = CelEngine()
    fn = build_load_skills_node(_make_ctx(node_def), cel, skill_registry, tool_registry)

    with patch(
        "sherma.langgraph.skill_tools.load_and_register_skill",
        return_value=("# Weather Skill\nUse get_weather tool.", ["get_weather"]),
    ) as mock_load:
        result = await fn({"messages": [], INTERNAL_STATE_KEY: {}})

    mock_load.assert_called_once()
    msgs = result["messages"]
    assert len(msgs) == 2
    # First message: AIMessage with tool_calls
    ai_msg = msgs[0]
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "load_skill_md"
    assert ai_msg.tool_calls[0]["args"]["skill_id"] == "weather"
    # Second message: ToolMessage with skill content
    tool_msg = msgs[1]
    assert isinstance(tool_msg, ToolMessage)
    assert "Weather Skill" in tool_msg.content
    # Internal state tracks tool IDs
    internal = result[INTERNAL_STATE_KEY]
    assert "get_weather" in internal["loaded_tools_from_skills"]


@pytest.mark.asyncio
async def test_build_load_skills_node_multiple_skills():
    """load_skills loads multiple skills in a single AIMessage."""
    from unittest.mock import patch

    from langchain_core.messages import ToolMessage

    from sherma.registry.base import RegistryEntry

    skill, _card, skill_registry, tool_registry = _make_skill_registries()
    await skill_registry.add(
        RegistryEntry(id="weather", version="1.0.0", instance=skill)
    )

    node_def = NodeDef(
        name="load",
        type="load_skills",
        args=LoadSkillsArgs(
            skill_ids=(
                '[{"id": "weather", "version": "1.0.0"},'
                ' {"id": "calendar", "version": "2.0.0"}]'
            )
        ),
    )
    cel = CelEngine()
    fn = build_load_skills_node(_make_ctx(node_def), cel, skill_registry, tool_registry)

    call_count = 0

    async def mock_load(sid, ver, sr, tr, hm=None):
        nonlocal call_count
        call_count += 1
        if sid == "weather":
            return "# Weather", ["get_weather"]
        return "# Calendar", ["create_event"]

    with patch(
        "sherma.langgraph.skill_tools.load_and_register_skill",
        side_effect=mock_load,
    ):
        result = await fn({"messages": [], INTERNAL_STATE_KEY: {}})

    assert call_count == 2
    msgs = result["messages"]
    # 1 AIMessage with 2 tool_calls + 2 ToolMessages
    assert len(msgs) == 3
    ai_msg = msgs[0]
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.tool_calls) == 2
    assert isinstance(msgs[1], ToolMessage)
    assert isinstance(msgs[2], ToolMessage)
    internal = result[INTERNAL_STATE_KEY]
    assert "get_weather" in internal["loaded_tools_from_skills"]
    assert "create_event" in internal["loaded_tools_from_skills"]


@pytest.mark.asyncio
async def test_build_load_skills_node_empty_list():
    """load_skills with empty list produces no messages."""
    from sherma.registry.skill import SkillRegistry
    from sherma.registry.tool import ToolRegistry

    node_def = NodeDef(
        name="load",
        type="load_skills",
        args=LoadSkillsArgs(skill_ids="[]"),
    )
    cel = CelEngine()
    fn = build_load_skills_node(
        _make_ctx(node_def), cel, SkillRegistry(), ToolRegistry()
    )

    result = await fn({"messages": [], INTERNAL_STATE_KEY: {}})
    assert result["messages"] == []
    assert result[INTERNAL_STATE_KEY]["loaded_tools_from_skills"] == []


@pytest.mark.asyncio
async def test_build_load_skills_node_preserves_existing_tool_ids():
    """load_skills appends to existing loaded_tools_from_skills."""
    from unittest.mock import patch

    from sherma.registry.base import RegistryEntry

    skill, _card, skill_registry, tool_registry = _make_skill_registries()
    await skill_registry.add(
        RegistryEntry(id="weather", version="1.0.0", instance=skill)
    )

    node_def = NodeDef(
        name="load",
        type="load_skills",
        args=LoadSkillsArgs(skill_ids='[{"id": "weather"}]'),
    )
    cel = CelEngine()
    fn = build_load_skills_node(_make_ctx(node_def), cel, skill_registry, tool_registry)

    with patch(
        "sherma.langgraph.skill_tools.load_and_register_skill",
        return_value=("# Weather", ["get_weather"]),
    ):
        result = await fn(
            {
                "messages": [],
                INTERNAL_STATE_KEY: {"loaded_tools_from_skills": ["existing_tool"]},
            }
        )

    internal = result[INTERNAL_STATE_KEY]
    assert "existing_tool" in internal["loaded_tools_from_skills"]
    assert "get_weather" in internal["loaded_tools_from_skills"]


@pytest.mark.asyncio
async def test_build_load_skills_node_default_version():
    """load_skills defaults version to '*' when not provided."""
    from unittest.mock import patch

    from sherma.registry.base import RegistryEntry

    skill, _card, skill_registry, tool_registry = _make_skill_registries()
    await skill_registry.add(
        RegistryEntry(id="weather", version="1.0.0", instance=skill)
    )

    node_def = NodeDef(
        name="load",
        type="load_skills",
        args=LoadSkillsArgs(skill_ids='[{"id": "weather"}]'),
    )
    cel = CelEngine()
    fn = build_load_skills_node(_make_ctx(node_def), cel, skill_registry, tool_registry)

    with patch(
        "sherma.langgraph.skill_tools.load_and_register_skill",
        return_value=("# Weather", ["get_weather"]),
    ) as mock_load:
        await fn({"messages": [], INTERNAL_STATE_KEY: {}})

    # version should default to "*"
    call_args = mock_load.call_args
    assert call_args[0][1] == "*"


@pytest.mark.asyncio
async def test_build_load_skills_node_skips_failed_skills():
    """load_skills continues loading when a skill fails."""
    from unittest.mock import patch

    from sherma.registry.base import RegistryEntry

    skill, _card, skill_registry, tool_registry = _make_skill_registries()
    await skill_registry.add(
        RegistryEntry(id="weather", version="1.0.0", instance=skill)
    )

    node_def = NodeDef(
        name="load",
        type="load_skills",
        args=LoadSkillsArgs(
            skill_ids='[{"id": "broken"}, {"id": "weather", "version": "1.0.0"}]'
        ),
    )
    cel = CelEngine()
    fn = build_load_skills_node(_make_ctx(node_def), cel, skill_registry, tool_registry)

    async def mock_load(sid, ver, sr, tr, hm=None):
        if sid == "broken":
            raise RuntimeError("Skill not found")
        return "# Weather", ["get_weather"]

    with patch(
        "sherma.langgraph.skill_tools.load_and_register_skill",
        side_effect=mock_load,
    ):
        result = await fn({"messages": [], INTERNAL_STATE_KEY: {}})

    # Only the working skill should produce messages
    msgs = result["messages"]
    assert len(msgs) == 2  # 1 AIMessage + 1 ToolMessage
    internal = result[INTERNAL_STATE_KEY]
    assert "get_weather" in internal["loaded_tools_from_skills"]


# --- Custom node tests ---


@pytest.mark.asyncio
async def test_build_custom_node_no_hooks():
    """Custom node without hooks returns empty dict."""
    from sherma.langgraph.declarative.nodes import build_custom_node
    from sherma.langgraph.declarative.schema import CustomArgs

    node_def = NodeDef(
        name="my_custom",
        type="custom",
        args=CustomArgs(),
    )
    fn = build_custom_node(_make_ctx(node_def))
    result = await fn({"counter": 5})
    assert result == {}


@pytest.mark.asyncio
async def test_build_custom_node_with_hook():
    """Custom node with node_execute hook returns hook-provided result."""
    from sherma.hooks.types import NodeExecuteContext
    from sherma.langgraph.declarative.nodes import build_custom_node
    from sherma.langgraph.declarative.schema import CustomArgs

    class MyExecuteHook(BaseHookExecutor):
        async def node_execute(
            self, ctx: NodeExecuteContext
        ) -> NodeExecuteContext | None:
            ctx.result = {"doubled": ctx.state["counter"] * 2}
            return ctx

    hook_manager = HookManager()
    hook_manager.register(MyExecuteHook())

    node_def = NodeDef(
        name="doubler",
        type="custom",
        args=CustomArgs(),
    )
    fn = build_custom_node(_make_ctx(node_def, hook_manager=hook_manager))
    result = await fn({"counter": 5})
    assert result == {"doubled": 10}


@pytest.mark.asyncio
async def test_custom_node_fires_all_hooks():
    """Custom node fires node_enter, node_execute, node_exit in order."""
    from sherma.hooks.types import NodeExecuteContext
    from sherma.langgraph.declarative.nodes import build_custom_node
    from sherma.langgraph.declarative.schema import CustomArgs

    events: list[str] = []

    class TrackingHook(BaseHookExecutor):
        async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
            events.append("node_enter")
            return None

        async def node_execute(
            self, ctx: NodeExecuteContext
        ) -> NodeExecuteContext | None:
            events.append("node_execute")
            ctx.result = {"done": True}
            return ctx

        async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
            events.append("node_exit")
            return None

    hook_manager = HookManager()
    hook_manager.register(TrackingHook())

    node_def = NodeDef(
        name="my_custom",
        type="custom",
        args=CustomArgs(),
    )
    fn = build_custom_node(_make_ctx(node_def, hook_manager=hook_manager))
    result = await fn({})
    assert events == ["node_enter", "node_execute", "node_exit"]
    assert result == {"done": True}


@pytest.mark.asyncio
async def test_custom_node_exit_can_modify_result():
    """node_exit hook can modify the result produced by node_execute."""
    from sherma.hooks.types import NodeExecuteContext
    from sherma.langgraph.declarative.nodes import build_custom_node
    from sherma.langgraph.declarative.schema import CustomArgs

    class ExecuteHook(BaseHookExecutor):
        async def node_execute(
            self, ctx: NodeExecuteContext
        ) -> NodeExecuteContext | None:
            ctx.result = {"value": 1}
            return ctx

        async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
            ctx.result["extra"] = True
            return ctx

    hook_manager = HookManager()
    hook_manager.register(ExecuteHook())

    node_def = NodeDef(
        name="custom_with_exit",
        type="custom",
        args=CustomArgs(),
    )
    fn = build_custom_node(_make_ctx(node_def, hook_manager=hook_manager))
    result = await fn({})
    assert result == {"value": 1, "extra": True}


@pytest.mark.asyncio
async def test_custom_node_dispatches_by_name():
    """Multiple custom nodes with a single hook that dispatches by node_name."""
    from sherma.hooks.types import NodeExecuteContext
    from sherma.langgraph.declarative.nodes import build_custom_node
    from sherma.langgraph.declarative.schema import CustomArgs

    class MultiHook(BaseHookExecutor):
        async def node_execute(
            self, ctx: NodeExecuteContext
        ) -> NodeExecuteContext | None:
            if ctx.node_name == "add_one":
                ctx.result = {"value": ctx.state.get("value", 0) + 1}
            elif ctx.node_name == "double":
                ctx.result = {"value": ctx.state.get("value", 0) * 2}
            return ctx

    hook_manager = HookManager()
    hook_manager.register(MultiHook())

    node_a = NodeDef(name="add_one", type="custom", args=CustomArgs())
    node_b = NodeDef(name="double", type="custom", args=CustomArgs())

    fn_a = build_custom_node(_make_ctx(node_a, hook_manager=hook_manager))
    fn_b = build_custom_node(_make_ctx(node_b, hook_manager=hook_manager))

    assert await fn_a({"value": 5}) == {"value": 6}
    assert await fn_b({"value": 5}) == {"value": 10}


@pytest.mark.asyncio
async def test_custom_node_metadata_accessible():
    """Hook can access metadata from CustomArgs via node_context."""
    from sherma.hooks.types import NodeExecuteContext
    from sherma.langgraph.declarative.nodes import build_custom_node
    from sherma.langgraph.declarative.schema import CustomArgs

    class MetadataHook(BaseHookExecutor):
        async def node_execute(
            self, ctx: NodeExecuteContext
        ) -> NodeExecuteContext | None:
            meta = ctx.node_context.node_def.args.metadata  # type: ignore[union-attr]
            ctx.result = {"greeting": meta.get("prefix", "") + " world"}
            return ctx

    hook_manager = HookManager()
    hook_manager.register(MetadataHook())

    node_def = NodeDef(
        name="greet",
        type="custom",
        args=CustomArgs(metadata={"prefix": "hello"}),
    )
    fn = build_custom_node(_make_ctx(node_def, hook_manager=hook_manager))
    result = await fn({})
    assert result == {"greeting": "hello world"}


@pytest.mark.asyncio
async def test_custom_node_error_handling():
    """Custom node fires on_node_error when node_execute hook raises."""
    from sherma.hooks.types import NodeExecuteContext, OnNodeErrorContext
    from sherma.langgraph.declarative.nodes import build_custom_node
    from sherma.langgraph.declarative.schema import CustomArgs

    class FailingHook(BaseHookExecutor):
        async def node_execute(
            self, ctx: NodeExecuteContext
        ) -> NodeExecuteContext | None:
            raise ValueError("custom node failed")

    class ConsumeErrorHook(BaseHookExecutor):
        async def on_node_error(
            self, ctx: OnNodeErrorContext
        ) -> OnNodeErrorContext | None:
            ctx.error = None
            return ctx

    hook_manager = HookManager()
    hook_manager.register(FailingHook())
    hook_manager.register(ConsumeErrorHook())

    node_def = NodeDef(
        name="failing_custom",
        type="custom",
        args=CustomArgs(),
    )
    fn = build_custom_node(_make_ctx(node_def, hook_manager=hook_manager))
    result = await fn({})
    assert result == {}


@pytest.mark.asyncio
async def test_custom_node_registries_exposed_on_exec_ctx():
    """registries supplied via NodeContext are forwarded to NodeExecuteContext."""
    from sherma.hooks.types import NodeExecuteContext
    from sherma.langgraph.declarative.nodes import build_custom_node
    from sherma.langgraph.declarative.schema import CustomArgs
    from sherma.registry.bundle import RegistryBundle

    seen: dict[str, RegistryBundle | None] = {"registries": None}

    class CaptureHook(BaseHookExecutor):
        async def node_execute(
            self, ctx: NodeExecuteContext
        ) -> NodeExecuteContext | None:
            seen["registries"] = ctx.registries
            ctx.result = {"ok": True}
            return ctx

    hook_manager = HookManager()
    hook_manager.register(CaptureHook())

    registries = RegistryBundle(tenant_id="test-tenant")
    node_def = NodeDef(name="capture", type="custom", args=CustomArgs())
    ctx = NodeContext(
        config=DeclarativeConfig(manifest_version=1),
        node_def=node_def,
        hook_manager=hook_manager,
        registries=registries,
    )
    fn = build_custom_node(ctx)
    result = await fn({})

    assert result == {"ok": True}
    assert seen["registries"] is registries


@pytest.mark.asyncio
async def test_custom_node_registries_default_none_when_unset():
    """registries defaults to None when the NodeContext omits it."""
    from sherma.hooks.types import NodeExecuteContext
    from sherma.langgraph.declarative.nodes import build_custom_node
    from sherma.langgraph.declarative.schema import CustomArgs

    seen: dict[str, object] = {"registries": "sentinel"}

    class CaptureHook(BaseHookExecutor):
        async def node_execute(
            self, ctx: NodeExecuteContext
        ) -> NodeExecuteContext | None:
            seen["registries"] = ctx.registries
            return ctx

    hook_manager = HookManager()
    hook_manager.register(CaptureHook())

    node_def = NodeDef(name="no_reg", type="custom", args=CustomArgs())
    fn = build_custom_node(_make_ctx(node_def, hook_manager=hook_manager))
    await fn({})

    assert seen["registries"] is None


@pytest.mark.asyncio
async def test_custom_node_hook_uses_chat_model_via_registries():
    """node_execute hook resolves a chat model through ctx.registries."""
    from sherma.hooks.types import NodeExecuteContext
    from sherma.langgraph.declarative.nodes import build_custom_node
    from sherma.langgraph.declarative.schema import CustomArgs
    from sherma.registry.bundle import RegistryBundle

    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="summarised"))

    class SummarizeHook(BaseHookExecutor):
        async def node_execute(
            self, ctx: NodeExecuteContext
        ) -> NodeExecuteContext | None:
            assert ctx.registries is not None
            model = ctx.registries.chat_models["summarizer"]
            response = await model.ainvoke(ctx.state["messages"])
            ctx.result = {"summary": response.content}
            return ctx

    hook_manager = HookManager()
    hook_manager.register(SummarizeHook())

    registries = RegistryBundle(chat_models={"summarizer": chat_model})
    node_def = NodeDef(name="summarize", type="custom", args=CustomArgs())
    ctx = NodeContext(
        config=DeclarativeConfig(manifest_version=1),
        node_def=node_def,
        hook_manager=hook_manager,
        registries=registries,
    )
    fn = build_custom_node(ctx)
    result = await fn({"messages": ["hi", "there"]})

    assert result == {"summary": "summarised"}
    chat_model.ainvoke.assert_awaited_once_with(["hi", "there"])


@pytest.mark.asyncio
async def test_custom_node_hook_uses_tool_registry_via_registries():
    """node_execute hook resolves a tool through ctx.registries.tool_registry."""
    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.hooks.types import NodeExecuteContext
    from sherma.langgraph.declarative.nodes import build_custom_node
    from sherma.langgraph.declarative.schema import CustomArgs
    from sherma.registry.base import RegistryEntry
    from sherma.registry.bundle import RegistryBundle

    def echo(text: str) -> str:
        """Echo the input."""
        return text

    lc_tool = StructuredTool.from_function(func=echo, name="echo")
    tool_entity = Tool(id="echo", version="1.0.0", function=lc_tool)

    class LookupHook(BaseHookExecutor):
        async def node_execute(
            self, ctx: NodeExecuteContext
        ) -> NodeExecuteContext | None:
            assert ctx.registries is not None
            tool = await ctx.registries.tool_registry.get("echo", "==1.0.0")
            ctx.result = {"tool_id": tool.id, "tool_version": tool.version}
            return ctx

    hook_manager = HookManager()
    hook_manager.register(LookupHook())

    registries = RegistryBundle()
    await registries.tool_registry.add(
        RegistryEntry(id="echo", version="1.0.0", instance=tool_entity)
    )

    node_def = NodeDef(name="lookup", type="custom", args=CustomArgs())
    ctx = NodeContext(
        config=DeclarativeConfig(manifest_version=1),
        node_def=node_def,
        hook_manager=hook_manager,
        registries=registries,
    )
    fn = build_custom_node(ctx)
    result = await fn({})

    assert result == {"tool_id": "echo", "tool_version": "1.0.0"}


# ---------------------------------------------------------------------------
# state_updates for call_llm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_llm_state_updates_content_to_custom_field():
    """state_updates maps llm_response.content to a custom state field."""
    node_def = NodeDef(
        name="summarizer",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"Summarize"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            state_updates={"summary": "llm_response.content"},
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="This is a summary."))
    cel = CelEngine()

    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    result = await fn({"messages": [], "summary": ""})

    assert result == {"summary": "This is a summary."}
    assert "messages" not in result


@pytest.mark.asyncio
async def test_call_llm_state_updates_multiple_fields():
    """state_updates can map to multiple state fields at once."""
    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"Be helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            state_updates={
                "last_response": "llm_response.content",
                "call_count": "state.call_count + 1",
            },
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hi there"))
    cel = CelEngine()

    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    result = await fn({"messages": [], "last_response": "", "call_count": 0})

    assert result == {"last_response": "Hi there", "call_count": 1}


@pytest.mark.asyncio
async def test_call_llm_state_updates_tool_calls_extraction():
    """state_updates can extract tool_calls from the LLM response."""
    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"Be helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            state_updates={
                "last_tool_calls": "llm_response.tool_calls",
            },
        ),
    )
    tool_calls = [{"name": "get_weather", "args": {"city": "NYC"}, "id": "tc1"}]
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(
        return_value=AIMessage(content="", tool_calls=tool_calls)
    )
    cel = CelEngine()

    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    result = await fn({"messages": [], "last_tool_calls": []})

    assert len(result["last_tool_calls"]) == 1
    assert result["last_tool_calls"][0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_call_llm_state_updates_warns_tooled_node_missing_messages(caplog):
    """Emit a warning when state_updates omits messages on a tooled call_llm."""
    import logging

    from langchain_core.tools import StructuredTool

    from sherma.entities.tool import Tool
    from sherma.registry.base import RegistryEntry
    from sherma.registry.tool import ToolRegistry

    node_def = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt=[
                PromptMessageDef(role="system", content='"Be helpful"'),
                PromptMessageDef(role="messages", content="state.messages"),
            ],
            tools=[RegistryRef(id="my-tool", version="1.0.0")],
            state_updates={"summary": "llm_response.content"},
        ),
    )
    bound_model = AsyncMock()
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="Result"))
    chat_model = MagicMock()
    chat_model.bind_tools = MagicMock(return_value=bound_model)

    registry = ToolRegistry()
    real_tool = StructuredTool.from_function(
        func=lambda x: x, name="my-tool", description="test"
    )
    await registry.add(
        RegistryEntry(
            id="my-tool",
            version="1.0.0",
            instance=Tool(id="my-tool", version="1.0.0", function=real_tool),
        )
    )

    cel = CelEngine()
    fn = build_call_llm_node(
        _make_ctx(node_def), chat_model, cel, tool_registry=registry
    )

    with caplog.at_level(logging.WARNING):
        result = await fn({"messages": [], "summary": ""})

    assert result == {"summary": "Result"}
    assert "does not include 'messages'" in caplog.text
