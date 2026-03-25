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
    build_set_state_node,
    build_tool_node,
)
from sherma.langgraph.declarative.schema import (
    CallLLMArgs,
    DataTransformArgs,
    DeclarativeConfig,
    InterruptArgs,
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
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))
    cel = CelEngine()

    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    result = await fn({"messages": []})

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Hello!"
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
    assert result["messages"][0].content == "Using tool"


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
    assert result["messages"][0].content == "Using skill tool"


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
    assert result["messages"][0].content == "merged"


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

    assert result["messages"][0].content == "No tools"
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
    assert result["messages"][0].content == "Using registry tools"


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

    # Dict response should be wrapped as AIMessage with JSON content
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    import json

    assert json.loads(msg.content) == {"name": "Alice"}


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

    # AIMessage response should pass through without wrapping
    assert result["messages"][0] is ai_response


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
        ),
    )
    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))
    cel = CelEngine()

    fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
    result = await fn({"messages": []})
    assert result["messages"][0].content == "Hello!"


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
    assert result["messages"][0].content == "Using sub-agents"


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
    assert result["messages"][0].content == "Subset"


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
        ),
    )

    chat_model = AsyncMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="No tools"))

    cel = CelEngine()
    ctx = _make_ctx(node_def)
    ctx.extra["sub_agent_tool_ids"] = ["weather-agent"]
    fn = build_call_llm_node(ctx, chat_model, cel)

    result = await fn({"messages": []})

    assert result["messages"][0].content == "No tools"
    # bind_tools should NOT have been called since it's an AsyncMock (no bind_tools)
    chat_model.ainvoke.assert_called_once()
