"""Tests for custom CEL functions (json, jsonValid, default, string extensions)."""

from __future__ import annotations

import pytest

from sherma.exceptions import CelEvaluationError
from sherma.langgraph.declarative.cel_engine import CelEngine

# ---------------------------------------------------------------------------
# Tier 1: json()
# ---------------------------------------------------------------------------


class TestJson:
    def test_parse_object(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('json(\'{"status": "ok", "count": 3}\')', {})
        assert result == {"status": "ok", "count": 3}

    def test_parse_array(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("json('[1, 2, 3]')", {})
        assert result == [1, 2, 3]

    def test_parse_nested(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('json(\'{"a": {"b": [1, 2]}}\')["a"]["b"][0]', {})
        assert result == 1

    def test_field_access(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('json(\'{"name": "alice"}\')["name"]', {})
        assert result == "alice"

    def test_parse_from_state(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'json(state.data)["key"]',
            {"data": '{"key": "value"}'},
        )
        assert result == "value"

    def test_invalid_json_raises(self) -> None:
        cel = CelEngine()
        with pytest.raises(CelEvaluationError, match="CEL evaluation failed"):
            cel.evaluate("json('not json')", {})

    def test_parse_string_primitive(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("json('\"hello\"')", {})
        assert result == "hello"

    def test_parse_number_primitive(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("json('42')", {})
        assert result == 42

    def test_parse_boolean_primitive(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("json('true')", {})
        assert result is True

    def test_method_style(self) -> None:
        """json() can be called as a method on a string."""
        cel = CelEngine()
        result = cel.evaluate(
            'state.data.json()["status"]',
            {"data": '{"status": "done"}'},
        )
        assert result == "done"

    def test_with_message_content(self) -> None:
        """Parse JSON from a LangChain message content field."""
        from langchain_core.messages import AIMessage

        cel = CelEngine()
        msg = AIMessage(content='{"action": "escalate", "reason": "complex"}')
        result = cel.evaluate(
            'json(state.messages[0]["content"])["action"]',
            {"messages": [msg]},
        )
        assert result == "escalate"


# ---------------------------------------------------------------------------
# Tier 1: jsonValid()
# ---------------------------------------------------------------------------


class TestJsonValid:
    def test_valid_object(self) -> None:
        cel = CelEngine()
        assert cel.evaluate_bool("jsonValid('{\"a\": 1}')", {}) is True

    def test_valid_array(self) -> None:
        cel = CelEngine()
        assert cel.evaluate_bool("jsonValid('[1, 2]')", {}) is True

    def test_invalid(self) -> None:
        cel = CelEngine()
        assert cel.evaluate_bool("jsonValid('not json')", {}) is False

    def test_empty_string(self) -> None:
        cel = CelEngine()
        assert cel.evaluate_bool("jsonValid('')", {}) is False

    def test_from_state(self) -> None:
        cel = CelEngine()
        assert (
            cel.evaluate_bool("jsonValid(state.data)", {"data": '{"ok": true}'}) is True
        )


# ---------------------------------------------------------------------------
# Tier 2: default()
# ---------------------------------------------------------------------------


class TestDefault:
    def test_returns_value_when_no_error(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("default(state.x, 0)", {"x": 42})
        assert result == 42

    def test_returns_fallback_on_missing_key(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('default(state.missing, "none")', {})
        assert result == "none"

    def test_returns_fallback_on_json_error(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'default(json(state.data)["key"], "fallback")',
            {"data": "not json"},
        )
        assert result == "fallback"

    def test_returns_fallback_on_nested_error(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'default(json(state.data)["missing"], "default_val")',
            {"data": '{"other": 1}'},
        )
        assert result == "default_val"

    def test_with_integer_fallback(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("default(state.count, 0)", {})
        assert result == 0

    def test_with_complex_expression(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'default(json(state.body)["items"][0]["name"], "unknown")',
            {"body": '{"items": [{"name": "first"}]}'},
        )
        assert result == "first"

    def test_fallback_also_errors_raises(self) -> None:
        cel = CelEngine()
        with pytest.raises(CelEvaluationError):
            cel.evaluate("default(state.missing, state.also_missing)", {})

    def test_with_boolean_result(self) -> None:
        cel = CelEngine()
        result = cel.evaluate_bool("default(state.flag, false)", {"flag": True})
        assert result is True

    def test_with_boolean_fallback(self) -> None:
        cel = CelEngine()
        result = cel.evaluate_bool("default(state.missing, true)", {})
        assert result is True


# ---------------------------------------------------------------------------
# Tier 3: String extensions
# ---------------------------------------------------------------------------


class TestSplit:
    def test_basic(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"a,b,c".split(",")', {})
        assert result == ["a", "b", "c"]

    def test_from_state(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('state.tags.split(",")', {"tags": "x,y,z"})
        assert result == ["x", "y", "z"]

    def test_no_match(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"hello".split(",")', {})
        assert result == ["hello"]


class TestTrim:
    def test_basic(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"  hello  ".trim()', {})
        assert result == "hello"

    def test_no_whitespace(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"hello".trim()', {})
        assert result == "hello"

    def test_from_state(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("state.input.trim()", {"input": "\n data \t"})
        assert result == "data"


class TestLowerAscii:
    def test_basic(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"HELLO".lowerAscii()', {})
        assert result == "hello"

    def test_mixed(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"Hello World".lowerAscii()', {})
        assert result == "hello world"


class TestUpperAscii:
    def test_basic(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"hello".upperAscii()', {})
        assert result == "HELLO"


class TestReplace:
    def test_basic(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"hello world".replace("world", "CEL")', {})
        assert result == "hello CEL"

    def test_multiple_occurrences(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"aaa".replace("a", "b")', {})
        assert result == "bbb"

    def test_no_match(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"hello".replace("xyz", "abc")', {})
        assert result == "hello"


class TestIndexOf:
    def test_found(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"hello".indexOf("ll")', {})
        assert result == 2

    def test_not_found(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"hello".indexOf("xyz")', {})
        assert result == -1

    def test_at_start(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"hello".indexOf("he")', {})
        assert result == 0


class TestJoin:
    def test_basic(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('["a", "b", "c"].join(", ")', {})
        assert result == "a, b, c"

    def test_empty_separator(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('["a", "b", "c"].join("")', {})
        assert result == "abc"

    def test_from_state(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('state.items.join("-")', {"items": ["x", "y"]})
        assert result == "x-y"


class TestSubstring:
    def test_basic(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"hello world".substring(0, 5)', {})
        assert result == "hello"

    def test_middle(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"hello world".substring(6, 11)', {})
        assert result == "world"

    def test_from_state(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("state.text.substring(0, 3)", {"text": "abcdef"})
        assert result == "abc"


# ---------------------------------------------------------------------------
# Tier 4: template()
# ---------------------------------------------------------------------------


class TestTemplate:
    def test_basic_substitution(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('template("Hello ${name}!", {"name": "world"})', {})
        assert result == "Hello world!"

    def test_multiple_placeholders(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'template("${greeting}, ${name}!", {"greeting": "Hi", "name": "Alice"})',
            {},
        )
        assert result == "Hi, Alice!"

    def test_unresolved_placeholders_left_as_is(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'template("Hello ${name}, your role is ${role}.", {"name": "Bob"})',
            {},
        )
        assert result == "Hello Bob, your role is ${role}."

    def test_empty_map(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('template("Hello ${name}!", {})', {})
        assert result == "Hello ${name}!"

    def test_non_string_values(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'template("count=${count}, flag=${flag}", {"count": 42, "flag": true})',
            {},
        )
        assert result == "count=42, flag=True"

    def test_empty_string_template(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('template("", {"key": "value"})', {})
        assert result == ""

    def test_repeated_placeholder(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'template("${x} and ${x}", {"x": "same"})',
            {},
        )
        assert result == "same and same"

    def test_with_state_values(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'template("Welcome ${user}!", {"user": state.username})',
            {"username": "Charlie"},
        )
        assert result == "Welcome Charlie!"

    def test_multiline_template(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'template(state.tpl, {"a": "first", "b": "second"})',
            {"tpl": "Line1: ${a}\nLine2: ${b}"},
        )
        assert result == "Line1: first\nLine2: second"


# ---------------------------------------------------------------------------
# Tier 5: List macros (built-in) and last()
# ---------------------------------------------------------------------------


class TestListMacros:
    """Tests for built-in CEL macros: filter, exists, all, exists_one, map."""

    def test_filter_primitives(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            "state.items.filter(x, x > 2)", {"items": [1, 2, 3, 4, 5]}
        )
        assert result == [3, 4, 5]

    def test_filter_maps(self) -> None:
        cel = CelEngine()
        messages = [
            {"type": "human", "content": "hello"},
            {"type": "ai", "content": "hi"},
            {"type": "human", "content": "bye"},
        ]
        result = cel.evaluate(
            'state.messages.filter(m, m["type"] == "human")', {"messages": messages}
        )
        assert result == [
            {"type": "human", "content": "hello"},
            {"type": "human", "content": "bye"},
        ]

    def test_filter_empty_result(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("state.items.filter(x, x > 100)", {"items": [1, 2, 3]})
        assert result == []

    def test_filter_with_size(self) -> None:
        """The pattern from issue #40: size(filter(...)) > 0."""
        cel = CelEngine()
        messages = [
            {"type": "human", "content": "hello"},
            {"type": "ai", "content": "hi"},
        ]
        result = cel.evaluate_bool(
            'size(state.messages.filter(m, m["type"] == "human")) > 0',
            {"messages": messages},
        )
        assert result is True

    def test_filter_nested_access(self) -> None:
        cel = CelEngine()
        messages = [
            {"type": "human", "additional_kwargs": {"type": "approval_decision"}},
            {"type": "ai", "additional_kwargs": {"type": "response"}},
            {"type": "human", "additional_kwargs": {"type": "question"}},
        ]
        expr = (
            'state.messages.filter(m, m["additional_kwargs"]'
            '["type"] == "approval_decision")'
        )
        result = cel.evaluate(expr, {"messages": messages})
        assert len(result) == 1
        assert result[0]["additional_kwargs"]["type"] == "approval_decision"

    def test_exists_true(self) -> None:
        cel = CelEngine()
        result = cel.evaluate_bool(
            "state.items.exists(x, x > 3)", {"items": [1, 2, 3, 4]}
        )
        assert result is True

    def test_exists_false(self) -> None:
        cel = CelEngine()
        result = cel.evaluate_bool(
            "state.items.exists(x, x > 100)", {"items": [1, 2, 3]}
        )
        assert result is False

    def test_exists_on_maps(self) -> None:
        cel = CelEngine()
        messages = [
            {"type": "human", "content": "hi"},
            {"type": "ai", "content": "hello"},
        ]
        result = cel.evaluate_bool(
            'state.messages.exists(m, m["type"] == "ai")', {"messages": messages}
        )
        assert result is True

    def test_all_true(self) -> None:
        cel = CelEngine()
        result = cel.evaluate_bool("state.items.all(x, x > 0)", {"items": [1, 2, 3]})
        assert result is True

    def test_all_false(self) -> None:
        cel = CelEngine()
        result = cel.evaluate_bool("state.items.all(x, x > 2)", {"items": [1, 2, 3]})
        assert result is False

    def test_exists_one_true(self) -> None:
        cel = CelEngine()
        result = cel.evaluate_bool(
            "state.items.exists_one(x, x > 4)", {"items": [1, 2, 3, 4, 5]}
        )
        assert result is True

    def test_exists_one_false_multiple(self) -> None:
        cel = CelEngine()
        result = cel.evaluate_bool(
            "state.items.exists_one(x, x > 2)", {"items": [1, 2, 3, 4, 5]}
        )
        assert result is False

    def test_map_transform(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("state.items.map(x, x * 2)", {"items": [1, 2, 3]})
        assert result == [2, 4, 6]

    def test_map_extract_field(self) -> None:
        cel = CelEngine()
        messages = [
            {"type": "human", "content": "hello"},
            {"type": "ai", "content": "hi"},
        ]
        result = cel.evaluate(
            'state.messages.map(m, m["type"])', {"messages": messages}
        )
        assert result == ["human", "ai"]

    def test_filter_with_langchain_messages(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        cel = CelEngine()
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
            HumanMessage(content="bye"),
        ]
        result = cel.evaluate(
            'state.messages.filter(m, m["type"] == "human")',
            {"messages": messages},
        )
        assert len(result) == 2

    def test_exists_with_langchain_messages(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        cel = CelEngine()
        messages = [HumanMessage(content="hello"), AIMessage(content="hi")]
        result = cel.evaluate_bool(
            'state.messages.exists(m, m["type"] == "ai")',
            {"messages": messages},
        )
        assert result is True


class TestLast:
    def test_basic(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("last(state.items)", {"items": [1, 2, 3]})
        assert result == 3

    def test_single_element(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("last(state.items)", {"items": [42]})
        assert result == 42

    def test_empty_list_raises(self) -> None:
        cel = CelEngine()
        with pytest.raises(CelEvaluationError, match="CEL evaluation failed"):
            cel.evaluate("last(state.items)", {"items": []})

    def test_last_map(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            "last(state.items)",
            {"items": [{"a": 1}, {"a": 2}]},
        )
        assert result == {"a": 2}

    def test_find_last_pattern(self) -> None:
        """last() + filter() implements the findLast pattern."""
        cel = CelEngine()
        messages = [
            {"type": "human", "content": "first"},
            {"type": "ai", "content": "response"},
            {"type": "human", "content": "last_human"},
        ]
        result = cel.evaluate(
            'last(state.messages.filter(m, m["type"] == "human"))',
            {"messages": messages},
        )
        assert result == {"type": "human", "content": "last_human"}

    def test_find_last_with_default(self) -> None:
        """default(last(filter(...))["field"], fallback) pattern."""
        cel = CelEngine()
        messages = [
            {"type": "human", "content": "hello"},
            {"type": "ai", "content": "response"},
        ]
        result = cel.evaluate(
            'default(last(state.messages.filter(m, m["type"] == "ai"))["content"], "")',
            {"messages": messages},
        )
        assert result == "response"

    def test_find_last_with_default_empty_fallback(self) -> None:
        """default() falls back when filter returns empty and last() errors."""
        cel = CelEngine()
        messages = [{"type": "human", "content": "hello"}]
        expr = (
            "default(last(state.messages.filter("
            'm, m["type"] == "ai"))["content"], "none")'
        )
        result = cel.evaluate(expr, {"messages": messages})
        assert result == "none"

    def test_from_state(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("last(state.tags)", {"tags": ["a", "b", "c"]})
        assert result == "c"

    def test_method_style(self) -> None:
        """last() can be called as a method on a list."""
        cel = CelEngine()
        result = cel.evaluate("state.items.last()", {"items": [1, 2, 3]})
        assert result == 3


# ---------------------------------------------------------------------------
# Integration: combining functions
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_json_with_trim(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'json(state.data.trim())["status"]',
            {"data": '  {"status": "ok"}  '},
        )
        assert result == "ok"

    def test_json_field_with_lower(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'json(state.data)["NAME"].lowerAscii()',
            {"data": '{"NAME": "ALICE"}'},
        )
        assert result == "alice"

    def test_split_then_join(self) -> None:
        cel = CelEngine()
        result = cel.evaluate('"a,b,c".split(",").join(" ")', {})
        assert result == "a b c"

    def test_conditional_with_json_valid(self) -> None:
        cel = CelEngine()
        result = cel.evaluate_bool(
            'jsonValid(state.data) && json(state.data)["ok"] == true',
            {"data": '{"ok": true}'},
        )
        assert result is True

    def test_conditional_with_json_valid_invalid(self) -> None:
        cel = CelEngine()
        result = cel.evaluate_bool(
            'jsonValid(state.data) && json(state.data)["ok"] == true',
            {"data": "not json"},
        )
        assert result is False

    def test_default_with_json_in_routing(self) -> None:
        """Simulate routing: extract action from JSON, fallback to 'continue'."""
        cel = CelEngine()
        result = cel.evaluate(
            'default(json(state.response)["action"], "continue")',
            {"response": '{"action": "escalate"}'},
        )
        assert result == "escalate"

    def test_default_with_json_fallback_in_routing(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            'default(json(state.response)["action"], "continue")',
            {"response": "plain text response"},
        )
        assert result == "continue"

    def test_filter_last_default_routing(self) -> None:
        """Full findLast pattern from issue #40."""
        cel = CelEngine()
        messages = [
            {"type": "human", "additional_kwargs": {"type": "question"}},
            {"type": "ai", "additional_kwargs": {"type": "response"}},
            {
                "type": "human",
                "content": "approved",
                "additional_kwargs": {"type": "approval_decision"},
            },
            {"type": "ai", "additional_kwargs": {"type": "response"}},
        ]
        expr = (
            "default(last(state.messages.filter("
            'm, m["additional_kwargs"]["type"]'
            ' == "approval_decision"))["content"], "")'
        )
        result = cel.evaluate(expr, {"messages": messages})
        assert result == "approved"

    def test_filter_last_default_no_match(self) -> None:
        """findLast pattern falls back gracefully when no match."""
        cel = CelEngine()
        messages = [
            {"type": "human", "content": "hello"},
            {"type": "ai", "content": "hi"},
        ]
        expr = (
            "default(last(state.messages.filter("
            'm, m["additional_kwargs"]["type"]'
            ' == "approval_decision"))["content"], "")'
        )
        result = cel.evaluate(expr, {"messages": messages})
        assert result == ""

    def test_exists_for_routing_condition(self) -> None:
        """exists() as a routing condition."""
        cel = CelEngine()
        messages = [
            {"type": "human", "content": "hello"},
            {"type": "ai", "content": "hi", "tool_calls": [{"name": "search"}]},
        ]
        result = cel.evaluate_bool(
            'state.messages.exists(m, m["type"] == "ai" && size(m["tool_calls"]) > 0)',
            {"messages": messages},
        )
        assert result is True


# ---------------------------------------------------------------------------
# Tier 5: slice()
# ---------------------------------------------------------------------------


class TestSlice:
    def test_basic(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("[1, 2, 3, 4, 5].slice(1, 3)", {})
        assert result == [2, 3]

    def test_from_start(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("[1, 2, 3, 4, 5].slice(0, 3)", {})
        assert result == [1, 2, 3]

    def test_to_end(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            "state.items.slice(2, size(state.items))",
            {"items": [10, 20, 30, 40, 50]},
        )
        assert result == [30, 40, 50]

    def test_empty_result(self) -> None:
        cel = CelEngine()
        result = cel.evaluate("[1, 2, 3].slice(2, 2)", {})
        assert result == []

    def test_full_list(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            "state.items.slice(0, size(state.items))",
            {"items": ["a", "b", "c"]},
        )
        assert result == ["a", "b", "c"]

    def test_with_state_indices(self) -> None:
        cel = CelEngine()
        result = cel.evaluate(
            "state.messages.slice(state.cursor, size(state.messages))",
            {"messages": ["m0", "m1", "m2", "m3", "m4"], "cursor": 3},
        )
        assert result == ["m3", "m4"]

    def test_keep_last_n(self) -> None:
        """Simulate keeping last N messages: slice(size - N, size)."""
        cel = CelEngine()
        result = cel.evaluate(
            "state.msgs.slice(size(state.msgs) - 2, size(state.msgs))",
            {"msgs": ["a", "b", "c", "d", "e"]},
        )
        assert result == ["d", "e"]
