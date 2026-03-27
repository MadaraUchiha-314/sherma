"""Tests for hook context dataclass construction."""

from __future__ import annotations

from unittest.mock import MagicMock

from sherma.hooks.types import (
    AfterAgentCallContext,
    AfterInterruptContext,
    AfterLLMCallContext,
    AfterSkillLoadContext,
    AfterToolCallContext,
    BeforeAgentCallContext,
    BeforeInterruptContext,
    BeforeLLMCallContext,
    BeforeSkillLoadContext,
    BeforeToolCallContext,
    GraphInvokeContext,
    HookType,
    NodeEnterContext,
    NodeExitContext,
)


def _mock_node_context() -> MagicMock:
    return MagicMock()


def test_hook_type_enum_values():
    assert HookType.BEFORE_LLM_CALL.value == "before_llm_call"
    assert HookType.AFTER_LLM_CALL.value == "after_llm_call"
    assert HookType.NODE_ENTER.value == "node_enter"
    assert HookType.NODE_EXIT.value == "node_exit"
    assert HookType.NODE_EXECUTE.value == "node_execute"
    assert len(HookType) == 18


def test_before_llm_call_context():
    ctx = BeforeLLMCallContext(
        node_context=_mock_node_context(),
        node_name="agent",
        messages=[],
        system_prompt="hello",
        tools=[],
        state={},
    )
    assert ctx.node_name == "agent"
    assert ctx.system_prompt == "hello"


def test_after_llm_call_context():
    ctx = AfterLLMCallContext(
        node_context=_mock_node_context(),
        node_name="agent",
        response="resp",
        state={},
    )
    assert ctx.response == "resp"


def test_before_tool_call_context():
    ctx = BeforeToolCallContext(
        node_context=_mock_node_context(),
        node_name="tools",
        tool_calls=[{"name": "foo"}],
        tools=[],
        state={},
    )
    assert len(ctx.tool_calls) == 1


def test_after_tool_call_context():
    ctx = AfterToolCallContext(
        node_context=_mock_node_context(),
        node_name="tools",
        result={"messages": []},
        state={},
    )
    assert ctx.result == {"messages": []}


def test_before_agent_call_context():
    ctx = BeforeAgentCallContext(
        node_context=_mock_node_context(),
        node_name="delegate",
        input_value="hi",
        agent=MagicMock(),
        state={},
    )
    assert ctx.input_value == "hi"


def test_after_agent_call_context():
    ctx = AfterAgentCallContext(
        node_context=_mock_node_context(),
        node_name="delegate",
        result={"messages": []},
        state={},
    )
    assert ctx.node_name == "delegate"


def test_before_skill_load_context_none_node_context():
    ctx = BeforeSkillLoadContext(
        node_context=None,
        skill_id="weather",
        version="*",
    )
    assert ctx.node_context is None
    assert ctx.skill_id == "weather"


def test_after_skill_load_context():
    ctx = AfterSkillLoadContext(
        node_context=None,
        skill_id="weather",
        version="1.0.0",
        content="# Skill",
        tools_loaded=["get_weather"],
    )
    assert ctx.content == "# Skill"
    assert ctx.tools_loaded == ["get_weather"]


def test_after_skill_load_context_default_tools():
    ctx = AfterSkillLoadContext(
        node_context=None,
        skill_id="weather",
        version="*",
        content="content",
    )
    assert ctx.tools_loaded == []


def test_node_enter_context():
    ctx = NodeEnterContext(
        node_context=_mock_node_context(),
        node_name="agent",
        node_type="call_llm",
        state={"messages": []},
    )
    assert ctx.node_type == "call_llm"


def test_node_exit_context():
    ctx = NodeExitContext(
        node_context=_mock_node_context(),
        node_name="agent",
        node_type="call_llm",
        result={"messages": ["hi"]},
        state={},
    )
    assert ctx.result == {"messages": ["hi"]}


def test_before_interrupt_context():
    ctx = BeforeInterruptContext(
        node_context=_mock_node_context(),
        node_name="ask",
        value="question?",
        state={},
    )
    assert ctx.value == "question?"


def test_after_interrupt_context():
    ctx = AfterInterruptContext(
        node_context=_mock_node_context(),
        node_name="ask",
        value="question?",
        response="answer",
        state={},
    )
    assert ctx.response == "answer"


def test_graph_invoke_context():
    ctx = GraphInvokeContext(
        agent_id="agent-1",
        thread_id="t1",
        config={"recursion_limit": 25, "configurable": {"thread_id": "t1"}},
        input={"messages": []},
    )
    assert ctx.agent_id == "agent-1"
    assert ctx.thread_id == "t1"
    assert ctx.config["recursion_limit"] == 25
    assert ctx.input == {"messages": []}
