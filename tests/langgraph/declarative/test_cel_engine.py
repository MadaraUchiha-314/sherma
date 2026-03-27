"""Tests for CEL engine."""

from __future__ import annotations

import pytest

from sherma.exceptions import CelEvaluationError
from sherma.langgraph.declarative.cel_engine import CelEngine


def test_evaluate_simple_string():
    cel = CelEngine()
    result = cel.evaluate('"hello world"', {})
    assert result == "hello world"


def test_evaluate_arithmetic():
    cel = CelEngine()
    result = cel.evaluate("state.x + state.y", {"x": 10, "y": 20})
    assert result == 30


def test_evaluate_string_concat():
    cel = CelEngine()
    result = cel.evaluate('state.name + " world"', {"name": "hello"})
    assert result == "hello world"


def test_evaluate_list_access():
    cel = CelEngine()
    result = cel.evaluate("state.items[0]", {"items": ["first", "second"]})
    assert result == "first"


def test_evaluate_map_access():
    cel = CelEngine()
    result = cel.evaluate('state.data["key"]', {"data": {"key": "value"}})
    assert result == "value"


def test_evaluate_bool_true():
    cel = CelEngine()
    result = cel.evaluate_bool("state.x > 5", {"x": 10})
    assert result is True


def test_evaluate_bool_false():
    cel = CelEngine()
    result = cel.evaluate_bool("state.x > 5", {"x": 3})
    assert result is False


def test_evaluate_bool_type_error():
    cel = CelEngine()
    with pytest.raises(CelEvaluationError, match="expected bool"):
        cel.evaluate_bool('"not a bool"', {})


def test_evaluate_with_extra_vars():
    extra = {"prompts": {"sys": {"instructions": "be helpful"}}}
    cel = CelEngine(extra_vars=extra)
    result = cel.evaluate('prompts["sys"]["instructions"]', {})
    assert result == "be helpful"


def test_evaluate_nested_map():
    cel = CelEngine()
    result = cel.evaluate(
        'state.config["nested"]["value"]',
        {"config": {"nested": {"value": "deep"}}},
    )
    assert result == "deep"


def test_evaluate_size():
    cel = CelEngine()
    result = cel.evaluate_bool("size(state.items) > 0", {"items": ["a"]})
    assert result is True


def test_evaluate_size_empty():
    cel = CelEngine()
    result = cel.evaluate_bool("size(state.items) == 0", {"items": []})
    assert result is True


def test_evaluate_parse_error():
    cel = CelEngine()
    with pytest.raises(CelEvaluationError, match="CEL parse error"):
        cel.evaluate("invalid!@#$syntax(((", {})


def test_evaluate_bool_expression():
    cel = CelEngine()
    result = cel.evaluate("true", {})
    assert result is True


def test_evaluate_list_result():
    cel = CelEngine()
    result = cel.evaluate("[1, 2, 3]", {})
    assert result == [1, 2, 3]


def test_evaluate_message_content_field():
    """CEL can access .content on LangChain message objects."""
    from langchain_core.messages import AIMessage

    cel = CelEngine()
    messages = [AIMessage(content="TASK_COMPLETE: done")]
    result = cel.evaluate('state.messages[0]["content"]', {"messages": messages})
    assert result == "TASK_COMPLETE: done"


def test_evaluate_message_type_field():
    """CEL can access .type on LangChain message objects."""
    from langchain_core.messages import HumanMessage

    cel = CelEngine()
    result = cel.evaluate('state.msg["type"]', {"msg": HumanMessage(content="hi")})
    assert result == "human"


def test_evaluate_message_content_contains():
    """CEL can use .contains() on message content for routing."""
    from langchain_core.messages import AIMessage

    cel = CelEngine()
    messages = [AIMessage(content="TASK_COMPLETE: all done")]
    result = cel.evaluate_bool(
        'state.messages[0]["content"].contains("TASK_COMPLETE")',
        {"messages": messages},
    )
    assert result is True


def test_evaluate_message_tool_calls():
    """CEL can access tool_calls on AI messages."""
    from langchain_core.messages import AIMessage

    cel = CelEngine()
    msg = AIMessage(content="", tool_calls=[{"id": "1", "name": "foo", "args": {}}])
    result = cel.evaluate('size(state.msg["tool_calls"])', {"msg": msg})
    assert result == 1


def test_evaluate_message_no_tool_calls_key():
    """HumanMessage has no tool_calls key — CEL should not find it."""
    from langchain_core.messages import HumanMessage

    cel = CelEngine()
    result = cel.evaluate_bool(
        "!has(state.msg.tool_calls)", {"msg": HumanMessage(content="hi")}
    )
    assert result is True


def test_evaluate_ai_message_empty_tool_calls():
    """AIMessage with no tool calls has an empty tool_calls list."""
    from langchain_core.messages import AIMessage

    cel = CelEngine()
    result = cel.evaluate(
        'size(state.msg["tool_calls"])', {"msg": AIMessage(content="hello")}
    )
    assert result == 0


def test_evaluate_dataclass_as_map():
    """Dataclass objects are converted to CEL maps with field access."""
    import dataclasses

    @dataclasses.dataclass
    class Point:
        x: int
        y: int

    cel = CelEngine()
    result = cel.evaluate('state.p["x"] + state.p["y"]', {"p": Point(x=3, y=7)})
    assert result == 10


def test_evaluate_plain_object_as_map():
    """Plain objects with __dict__ are converted to CEL maps."""

    class Config:
        def __init__(self) -> None:
            self.name = "test"
            self.enabled = True

    cel = CelEngine()
    result = cel.evaluate('state.cfg["name"]', {"cfg": Config()})
    assert result == "test"


def test_evaluate_state_bracket_access():
    """State fields can be accessed via state["key"] syntax."""
    cel = CelEngine()
    result = cel.evaluate('state["x"] + state["y"]', {"x": 10, "y": 20})
    assert result == 30


def test_template_with_prompt_extra_vars():
    """template() works with prompt instructions from extra_vars."""
    prompt_text = (
        "You are a helpful assistant.\n\n"
        "## Available Skills\n"
        "${skill_instructions}\n\n"
        "## User Context\n"
        "${user_context}"
    )
    cel = CelEngine(extra_vars={"prompts": {"plan": {"instructions": prompt_text}}})
    result = cel.evaluate(
        'template(prompts["plan"]["instructions"], '
        '{"skill_instructions": state.skills, "user_context": state.ctx})',
        {"skills": "Use the weather tool.", "ctx": "Location: NYC"},
    )
    assert "Use the weather tool." in result
    assert "Location: NYC" in result
    assert "${skill_instructions}" not in result
    assert "${user_context}" not in result
