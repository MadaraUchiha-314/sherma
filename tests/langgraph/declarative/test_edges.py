"""Tests for declarative edge builders."""

from __future__ import annotations

from langgraph.graph import END

from sherma.langgraph.declarative.cel_engine import CelEngine
from sherma.langgraph.declarative.edges import build_conditional_router
from sherma.langgraph.declarative.schema import BranchDef, EdgeDef


def test_conditional_router_first_match():
    edge = EdgeDef(
        source="agent",
        branches=[
            BranchDef(condition="x > 10", target="high"),
            BranchDef(condition="x > 5", target="medium"),
        ],
        default="low",
    )
    cel = CelEngine()
    router, _path_map = build_conditional_router(edge, cel)

    result = router({"x": 15})
    assert result == "high"


def test_conditional_router_second_match():
    edge = EdgeDef(
        source="agent",
        branches=[
            BranchDef(condition="x > 10", target="high"),
            BranchDef(condition="x > 5", target="medium"),
        ],
        default="low",
    )
    cel = CelEngine()
    router, _path_map = build_conditional_router(edge, cel)

    result = router({"x": 7})
    assert result == "medium"


def test_conditional_router_default():
    edge = EdgeDef(
        source="agent",
        branches=[
            BranchDef(condition="x > 10", target="high"),
        ],
        default="low",
    )
    cel = CelEngine()
    router, _path_map = build_conditional_router(edge, cel)

    result = router({"x": 3})
    assert result == "low"


def test_conditional_router_default_end():
    edge = EdgeDef(
        source="agent",
        branches=[
            BranchDef(condition="x > 10", target="high"),
        ],
        default="__end__",
    )
    cel = CelEngine()
    router, path_map = build_conditional_router(edge, cel)

    result = router({"x": 3})
    assert result == "__end__"
    assert path_map["__end__"] == END


def test_conditional_router_no_default():
    edge = EdgeDef(
        source="agent",
        branches=[
            BranchDef(condition="false", target="never"),
        ],
    )
    cel = CelEngine()
    router, _path_map = build_conditional_router(edge, cel)

    result = router({})
    assert result == "__end__"


def test_path_map_contains_targets():
    edge = EdgeDef(
        source="agent",
        branches=[
            BranchDef(condition="true", target="tools"),
        ],
        default="done",
    )
    cel = CelEngine()
    _, path_map = build_conditional_router(edge, cel)

    assert "tools" in path_map
    assert "done" in path_map
