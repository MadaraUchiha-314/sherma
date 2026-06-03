"""Tests for CELHookExecutor — CEL-expression based hooks."""

from __future__ import annotations

import pytest

from sherma.hooks.cel_executor import (
    CEL_CAPABLE_HOOKS,
    CELHookExecutor,
    hook_supports_state_updates,
    writable_fields,
)
from sherma.hooks.executor import HookExecutor
from sherma.hooks.types import (
    BeforeLLMCallContext,
    NodeEnterContext,
    NodeExitContext,
)
from sherma.langgraph.declarative.cel_engine import CelEngine


def _make_executor(spec: dict) -> CELHookExecutor:
    return CELHookExecutor(spec, CelEngine())


def test_satisfies_hook_executor_protocol():
    ex = _make_executor({"node_enter": {"set": {"node_name": "node_name"}}})
    assert isinstance(ex, HookExecutor)


def test_cel_capable_hooks_excludes_on_chat_model_create():
    assert "on_chat_model_create" not in CEL_CAPABLE_HOOKS
    assert "before_llm_call" in CEL_CAPABLE_HOOKS
    assert "node_exit" in CEL_CAPABLE_HOOKS


def test_writable_fields_excludes_complex_and_state():
    fields = writable_fields("before_llm_call")
    assert fields == {"node_name", "system_prompt"}
    # messages/tools (complex), node_context (non-serializable), state excluded
    assert "messages" not in fields
    assert "state" not in fields
    assert "node_context" not in fields


def test_writable_fields_unknown_hook_is_empty():
    assert writable_fields("does_not_exist") == frozenset()


def test_hook_supports_state_updates():
    assert hook_supports_state_updates("node_exit") is True  # has result + state
    assert hook_supports_state_updates("before_llm_call") is True  # has state
    assert hook_supports_state_updates("after_graph_invoke") is True  # has result
    assert hook_supports_state_updates("before_skill_load") is False  # neither
    assert hook_supports_state_updates("unknown") is False


@pytest.mark.asyncio
async def test_absent_hook_is_noop():
    ex = _make_executor({"node_enter": {"set": {"node_name": '"x"'}}})
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="agent",
        messages=[],
        system_prompt="Hi",
        tools=[],
        state={},
    )
    # before_llm_call not in spec -> None (pass through)
    assert await ex.before_llm_call(ctx) is None


@pytest.mark.asyncio
async def test_set_modifies_field():
    ex = _make_executor(
        {"before_llm_call": {"set": {"system_prompt": 'system_prompt + " EXTRA"'}}}
    )
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="agent",
        messages=[],
        system_prompt="Hi",
        tools=[],
        state={},
    )
    out = await ex.before_llm_call(ctx)
    assert out is ctx
    assert out.system_prompt == "Hi EXTRA"


@pytest.mark.asyncio
async def test_when_guard_false_is_noop():
    ex = _make_executor(
        {
            "before_llm_call": {
                "when": 'node_name == "other"',
                "set": {"system_prompt": '"changed"'},
            }
        }
    )
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="agent",
        messages=[],
        system_prompt="Hi",
        tools=[],
        state={},
    )
    out = await ex.before_llm_call(ctx)
    assert out is None
    assert ctx.system_prompt == "Hi"  # untouched


@pytest.mark.asyncio
async def test_when_guard_true_applies():
    ex = _make_executor(
        {
            "before_llm_call": {
                "when": 'node_name == "agent"',
                "set": {"system_prompt": '"changed"'},
            }
        }
    )
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="agent",
        messages=[],
        system_prompt="Hi",
        tools=[],
        state={},
    )
    out = await ex.before_llm_call(ctx)
    assert out is ctx
    assert out.system_prompt == "changed"


@pytest.mark.asyncio
async def test_state_updates_merge_into_result_for_result_bearing_ctx():
    ex = _make_executor(
        {
            "node_exit": {
                "state_updates": {"count": "state.count + 1"},
            }
        }
    )
    ctx = NodeExitContext(
        node_context=None,
        node_name="n",
        node_type="custom",
        result={"messages": ["m"]},
        state={"count": 4},
    )
    out = await ex.node_exit(ctx)
    assert out is ctx
    # merged into result (the node's returned state delta), preserving existing
    assert out.result == {"messages": ["m"], "count": 5}


@pytest.mark.asyncio
async def test_state_updates_fall_back_to_state_for_stateonly_ctx():
    ex = _make_executor({"node_enter": {"state_updates": {"seen": '"yes"'}}})
    ctx = NodeEnterContext(
        node_context=None,
        node_name="n",
        node_type="call_llm",
        state={"existing": 1},
    )
    out = await ex.node_enter(ctx)
    assert out is ctx
    assert out.state == {"existing": 1, "seen": "yes"}


@pytest.mark.asyncio
async def test_reads_complex_field_metadata():
    """response/result message metadata is readable in CEL."""
    ex = _make_executor(
        {
            "node_exit": {
                "state_updates": {
                    "decision": (
                        'result["messages"][size(result["messages"]) - 1]'
                        '["additional_kwargs"]["decision"]'
                    )
                }
            }
        }
    )
    from langchain_core.messages import HumanMessage

    msg = HumanMessage(content="ok", additional_kwargs={"decision": "approve"})
    ctx = NodeExitContext(
        node_context=None,
        node_name="get_approval",
        node_type="interrupt",
        result={"messages": [msg]},
        state={},
    )
    out = await ex.node_exit(ctx)
    assert out.result["decision"] == "approve"


@pytest.mark.asyncio
async def test_extra_vars_available():
    ex = CELHookExecutor(
        {"node_enter": {"state_updates": {"label": 'data["k"]'}}},
        CelEngine(extra_vars={"data": {"k": "v"}}),
    )
    ctx = NodeEnterContext(
        node_context=None, node_name="n", node_type="call_llm", state={}
    )
    out = await ex.node_enter(ctx)
    assert out.state["label"] == "v"
