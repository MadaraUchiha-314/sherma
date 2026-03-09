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
    result = cel.evaluate("x + y", {"x": 10, "y": 20})
    assert result == 30


def test_evaluate_string_concat():
    cel = CelEngine()
    result = cel.evaluate('name + " world"', {"name": "hello"})
    assert result == "hello world"


def test_evaluate_list_access():
    cel = CelEngine()
    result = cel.evaluate("items[0]", {"items": ["first", "second"]})
    assert result == "first"


def test_evaluate_map_access():
    cel = CelEngine()
    result = cel.evaluate('data["key"]', {"data": {"key": "value"}})
    assert result == "value"


def test_evaluate_bool_true():
    cel = CelEngine()
    result = cel.evaluate_bool("x > 5", {"x": 10})
    assert result is True


def test_evaluate_bool_false():
    cel = CelEngine()
    result = cel.evaluate_bool("x > 5", {"x": 3})
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
        'config["nested"]["value"]',
        {"config": {"nested": {"value": "deep"}}},
    )
    assert result == "deep"


def test_evaluate_size():
    cel = CelEngine()
    result = cel.evaluate_bool("size(items) > 0", {"items": ["a"]})
    assert result is True


def test_evaluate_size_empty():
    cel = CelEngine()
    result = cel.evaluate_bool("size(items) == 0", {"items": []})
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
