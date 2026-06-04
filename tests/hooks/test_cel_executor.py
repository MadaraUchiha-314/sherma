"""Tests for CELHookExecutor — CEL-expression based hooks."""

from __future__ import annotations

import pytest

from sherma.exceptions import CelEvaluationError
from sherma.hooks.cel_executor import (
    CEL_CAPABLE_HOOKS,
    CELHookExecutor,
    writable_fields,
)
from sherma.hooks.executor import HookExecutor
from sherma.hooks.types import (
    BeforeLLMCallContext,
    NodeEnterContext,
    NodeExitContext,
)
from sherma.langgraph.declarative.cel_engine import CelEngine


def _make_executor(expressions: dict[str, str]) -> CELHookExecutor:
    return CELHookExecutor(expressions, CelEngine())


def test_satisfies_hook_executor_protocol():
    ex = _make_executor({"node_enter": "{}"})
    assert isinstance(ex, HookExecutor)


def test_cel_capable_hooks_excludes_on_chat_model_create():
    assert "on_chat_model_create" not in CEL_CAPABLE_HOOKS
    assert "before_llm_call" in CEL_CAPABLE_HOOKS
    assert "node_exit" in CEL_CAPABLE_HOOKS


def test_writable_fields_excludes_complex_and_error():
    fields = writable_fields("before_llm_call")
    # state is writable now; messages/tools (complex) and node_context are not
    assert fields == {"node_name", "system_prompt", "state"}
    assert "messages" not in fields
    assert "node_context" not in fields


def test_writable_fields_unknown_hook_is_empty():
    assert writable_fields("does_not_exist") == frozenset()


@pytest.mark.asyncio
async def test_absent_hook_is_noop():
    ex = _make_executor({"node_enter": '{"node_name": "x"}'})
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="agent",
        messages=[],
        system_prompt="Hi",
        tools=[],
        state={},
    )
    assert await ex.before_llm_call(ctx) is None


@pytest.mark.asyncio
async def test_empty_map_is_noop():
    ex = _make_executor({"before_llm_call": "{}"})
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
    assert ctx.system_prompt == "Hi"


@pytest.mark.asyncio
async def test_returns_delta_overwrites_scalar_field():
    ex = _make_executor(
        {"before_llm_call": '{"system_prompt": system_prompt + " EXTRA"}'}
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
async def test_conditional_guard_via_ternary():
    expr = 'node_name == "other" ? {"system_prompt": "changed"} : {}'
    ex = _make_executor({"before_llm_call": expr})
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="agent",
        messages=[],
        system_prompt="Hi",
        tools=[],
        state={},
    )
    # guard false -> empty map -> no-op
    out = await ex.before_llm_call(ctx)
    assert out is None
    assert ctx.system_prompt == "Hi"


@pytest.mark.asyncio
async def test_dict_field_merges_preserving_existing_keys():
    """Returning {"result": {...}} merges, keeping original result entries."""
    ex = _make_executor({"node_exit": '{"result": {"response_type": "approve"}}'})
    sentinel = object()  # stands in for a real (non-CEL-convertible) message
    ctx = NodeExitContext(
        node_context=None,
        node_name="n",
        node_type="interrupt",
        result={"messages": [sentinel]},
        state={},
    )
    out = await ex.node_exit(ctx)
    assert out is ctx
    # response_type added; original messages object preserved verbatim
    assert out.result["response_type"] == "approve"
    assert out.result["messages"] == [sentinel]


@pytest.mark.asyncio
async def test_state_field_merges():
    ex = _make_executor({"node_enter": '{"state": {"seen": "yes"}}'})
    ctx = NodeEnterContext(
        node_context=None,
        node_name="n",
        node_type="call_llm",
        state={"existing": 1},
    )
    out = await ex.node_enter(ctx)
    assert out.state == {"existing": 1, "seen": "yes"}


@pytest.mark.asyncio
async def test_reads_complex_field_metadata():
    """result message metadata is readable in CEL."""
    ex = _make_executor(
        {
            "node_exit": (
                '{"result": {"decision": '
                'result["messages"][size(result["messages"]) - 1]'
                '["additional_kwargs"]["decision"]}}'
            )
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
    # the original message object is preserved (merge, not replace)
    assert out.result["messages"] == [msg]


@pytest.mark.asyncio
async def test_rejects_non_writable_field():
    ex = _make_executor({"before_llm_call": '{"messages": []}'})
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="agent",
        messages=[],
        system_prompt="Hi",
        tools=[],
        state={},
    )
    with pytest.raises(CelEvaluationError, match="cannot set field 'messages'"):
        await ex.before_llm_call(ctx)


@pytest.mark.asyncio
async def test_rejects_non_map_result():
    ex = _make_executor({"before_llm_call": '"not a map"'})
    ctx = BeforeLLMCallContext(
        node_context=None,
        node_name="agent",
        messages=[],
        system_prompt="Hi",
        tools=[],
        state={},
    )
    with pytest.raises(CelEvaluationError, match="must evaluate to a map"):
        await ex.before_llm_call(ctx)


@pytest.mark.asyncio
async def test_extra_vars_available():
    ex = CELHookExecutor(
        {"node_enter": '{"state": {"label": data["k"]}}'},
        CelEngine(extra_vars={"data": {"k": "v"}}),
    )
    ctx = NodeEnterContext(
        node_context=None, node_name="n", node_type="call_llm", state={}
    )
    out = await ex.node_enter(ctx)
    assert out.state["label"] == "v"
