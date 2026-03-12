"""Tests for HookManager: chaining, None pass-through, ordering."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.manager import HookManager
from sherma.hooks.types import (
    AfterGraphInvokeContext,
    BeforeLLMCallContext,
    GraphInvokeContext,
    NodeEnterContext,
    NodeExitContext,
)


@pytest.mark.asyncio
async def test_run_hook_no_executors():
    """With no executors, context passes through unchanged."""
    manager = HookManager()
    nc = MagicMock()
    ctx = BeforeLLMCallContext(nc, "n", [], "original", [], {})
    result = await manager.run_hook("before_llm_call", ctx)
    assert result is ctx
    assert result.system_prompt == "original"


@pytest.mark.asyncio
async def test_run_hook_none_passthrough():
    """Executor returning None leaves context unchanged."""
    manager = HookManager()
    manager.register(BaseHookExecutor())
    nc = MagicMock()
    ctx = BeforeLLMCallContext(nc, "n", [], "original", [], {})
    result = await manager.run_hook("before_llm_call", ctx)
    assert result is ctx


@pytest.mark.asyncio
async def test_run_hook_modifies_context():
    """Executor returning modified context replaces it."""

    class ModifyHook(BaseHookExecutor):
        async def before_llm_call(
            self, ctx: BeforeLLMCallContext
        ) -> BeforeLLMCallContext | None:
            ctx.system_prompt = "modified"
            return ctx

    manager = HookManager()
    manager.register(ModifyHook())
    nc = MagicMock()
    ctx = BeforeLLMCallContext(nc, "n", [], "original", [], {})
    result = await manager.run_hook("before_llm_call", ctx)
    assert result.system_prompt == "modified"


@pytest.mark.asyncio
async def test_run_hook_chaining_order():
    """Multiple executors run in registration order, chaining results."""

    class AppendA(BaseHookExecutor):
        async def before_llm_call(
            self, ctx: BeforeLLMCallContext
        ) -> BeforeLLMCallContext | None:
            ctx.system_prompt += "_A"
            return ctx

    class AppendB(BaseHookExecutor):
        async def before_llm_call(
            self, ctx: BeforeLLMCallContext
        ) -> BeforeLLMCallContext | None:
            ctx.system_prompt += "_B"
            return ctx

    manager = HookManager()
    manager.register(AppendA())
    manager.register(AppendB())
    nc = MagicMock()
    ctx = BeforeLLMCallContext(nc, "n", [], "start", [], {})
    result = await manager.run_hook("before_llm_call", ctx)
    assert result.system_prompt == "start_A_B"


@pytest.mark.asyncio
async def test_run_hook_mixed_none_and_modify():
    """None-returning executors don't break the chain."""

    class NoOp(BaseHookExecutor):
        pass

    class Modify(BaseHookExecutor):
        async def before_llm_call(
            self, ctx: BeforeLLMCallContext
        ) -> BeforeLLMCallContext | None:
            ctx.system_prompt = "changed"
            return ctx

    manager = HookManager()
    manager.register(NoOp())
    manager.register(Modify())
    manager.register(NoOp())
    nc = MagicMock()
    ctx = BeforeLLMCallContext(nc, "n", [], "original", [], {})
    result = await manager.run_hook("before_llm_call", ctx)
    assert result.system_prompt == "changed"


@pytest.mark.asyncio
async def test_run_hook_node_enter():
    """node_enter hook fires correctly."""
    calls: list[str] = []

    class TrackEnter(BaseHookExecutor):
        async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
            calls.append(f"enter:{ctx.node_name}")
            return None

    manager = HookManager()
    manager.register(TrackEnter())
    nc = MagicMock()
    ctx = NodeEnterContext(nc, "agent", "call_llm", {})
    await manager.run_hook("node_enter", ctx)
    assert calls == ["enter:agent"]


@pytest.mark.asyncio
async def test_run_hook_node_exit_modifies_result():
    """node_exit hook can modify the result."""

    class ModifyResult(BaseHookExecutor):
        async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
            ctx.result["extra"] = "added"
            return ctx

    manager = HookManager()
    manager.register(ModifyResult())
    nc = MagicMock()
    ctx = NodeExitContext(nc, "agent", "call_llm", {"messages": []}, {})
    result = await manager.run_hook("node_exit", ctx)
    assert result.result["extra"] == "added"


@pytest.mark.asyncio
async def test_run_hook_before_graph_invoke():
    """before_graph_invoke hook can modify the config."""

    class SetRecursionLimit(BaseHookExecutor):
        async def before_graph_invoke(
            self, ctx: GraphInvokeContext
        ) -> GraphInvokeContext | None:
            ctx.config["recursion_limit"] = 50
            return ctx

    manager = HookManager()
    manager.register(SetRecursionLimit())
    ctx = GraphInvokeContext(
        agent_id="agent-1",
        thread_id="t1",
        config={"recursion_limit": 25, "configurable": {"thread_id": "t1"}},
        input={"messages": []},
    )
    result = await manager.run_hook("before_graph_invoke", ctx)
    assert result.config["recursion_limit"] == 50


@pytest.mark.asyncio
async def test_run_hook_after_graph_invoke():
    """after_graph_invoke hook can modify the result."""

    class ModifyResult(BaseHookExecutor):
        async def after_graph_invoke(
            self, ctx: AfterGraphInvokeContext
        ) -> AfterGraphInvokeContext | None:
            ctx.result["extra"] = "added_by_hook"
            return ctx

    manager = HookManager()
    manager.register(ModifyResult())
    ctx = AfterGraphInvokeContext(
        agent_id="agent-1",
        thread_id="t1",
        config={"recursion_limit": 25, "configurable": {"thread_id": "t1"}},
        input={"messages": []},
        result={"messages": []},
    )
    result = await manager.run_hook("after_graph_invoke", ctx)
    assert result.result["extra"] == "added_by_hook"


def test_register_multiple():
    """Register adds executors in order."""
    manager = HookManager()
    e1 = BaseHookExecutor()
    e2 = BaseHookExecutor()
    manager.register(e1)
    manager.register(e2)
    assert manager._executors == [e1, e2]
