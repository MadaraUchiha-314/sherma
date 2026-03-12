"""Tests for BaseHookExecutor default behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sherma.hooks.executor import BaseHookExecutor, HookExecutor
from sherma.hooks.types import (
    AfterAgentCallContext,
    AfterGraphInvokeContext,
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
    NodeEnterContext,
    NodeExitContext,
)


def test_base_hook_executor_is_hook_executor():
    """BaseHookExecutor satisfies the HookExecutor protocol."""
    executor = BaseHookExecutor()
    assert isinstance(executor, HookExecutor)


@pytest.mark.asyncio
async def test_all_methods_return_none():
    """All BaseHookExecutor methods return None by default."""
    executor = BaseHookExecutor()
    nc = MagicMock()

    assert (
        await executor.before_llm_call(BeforeLLMCallContext(nc, "n", [], "", [], {}))
        is None
    )
    assert await executor.after_llm_call(AfterLLMCallContext(nc, "n", None, {})) is None
    assert (
        await executor.before_tool_call(BeforeToolCallContext(nc, "n", [], [], {}))
        is None
    )
    assert await executor.after_tool_call(AfterToolCallContext(nc, "n", {}, {})) is None
    assert (
        await executor.before_agent_call(
            BeforeAgentCallContext(nc, "n", "", MagicMock(), {})
        )
        is None
    )
    assert (
        await executor.after_agent_call(AfterAgentCallContext(nc, "n", {}, {})) is None
    )
    assert (
        await executor.before_skill_load(BeforeSkillLoadContext(None, "s", "*")) is None
    )
    assert (
        await executor.after_skill_load(AfterSkillLoadContext(None, "s", "*", ""))
        is None
    )
    assert await executor.node_enter(NodeEnterContext(nc, "n", "call_llm", {})) is None
    assert (
        await executor.node_exit(NodeExitContext(nc, "n", "call_llm", {}, {})) is None
    )
    assert (
        await executor.before_interrupt(BeforeInterruptContext(nc, "n", "", {})) is None
    )
    assert (
        await executor.after_interrupt(AfterInterruptContext(nc, "n", "", "", {}))
        is None
    )
    assert (
        await executor.before_graph_invoke(
            GraphInvokeContext("agent-1", "t1", {}, {"messages": []})
        )
        is None
    )
    assert (
        await executor.after_graph_invoke(
            AfterGraphInvokeContext("agent-1", "t1", {}, {"messages": []}, {})
        )
        is None
    )


@pytest.mark.asyncio
async def test_subclass_override():
    """Subclassing BaseHookExecutor and overriding a method works."""

    class MyHook(BaseHookExecutor):
        async def before_llm_call(
            self, ctx: BeforeLLMCallContext
        ) -> BeforeLLMCallContext | None:
            ctx.system_prompt = "modified"
            return ctx

    executor = MyHook()
    nc = MagicMock()
    ctx = BeforeLLMCallContext(nc, "n", [], "original", [], {})
    result = await executor.before_llm_call(ctx)
    assert result is not None
    assert result.system_prompt == "modified"
