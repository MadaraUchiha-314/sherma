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
