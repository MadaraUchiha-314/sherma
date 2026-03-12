"""Tests for error handling hooks (on_node_error and on_error)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.manager import HookManager
from sherma.hooks.types import OnErrorContext, OnNodeErrorContext

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeNodeDef:
    name: str = "test_node"
    type: str = "call_llm"
    args: Any = None


@dataclass
class FakeNodeContext:
    """Minimal stand-in for NodeContext without importing heavy deps."""

    config: Any = None
    node_def: Any = None
    extra: dict[str, Any] = field(default_factory=dict)
    hook_manager: HookManager | None = None


# ---------------------------------------------------------------------------
# on_node_error tests
# ---------------------------------------------------------------------------


class _ConsumeErrorExecutor(BaseHookExecutor):
    """Hook that consumes (swallows) the error."""

    async def on_node_error(self, ctx: OnNodeErrorContext) -> OnNodeErrorContext | None:
        ctx.error = None
        return ctx


class _ReplaceErrorExecutor(BaseHookExecutor):
    """Hook that replaces the error with a ValueError."""

    async def on_node_error(self, ctx: OnNodeErrorContext) -> OnNodeErrorContext | None:
        ctx.error = ValueError("replaced")
        return ctx


class _TrackingExecutor(BaseHookExecutor):
    """Hook that records calls without modifying them."""

    def __init__(self) -> None:
        self.calls: list[OnNodeErrorContext] = []

    async def on_node_error(self, ctx: OnNodeErrorContext) -> OnNodeErrorContext | None:
        self.calls.append(ctx)
        return None


@pytest.mark.asyncio
async def test_on_node_error_fires_when_node_raises() -> None:
    """on_node_error hook receives the exception from a failing node."""
    tracker = _TrackingExecutor()
    manager = HookManager()
    manager.register(tracker)

    original_err = RuntimeError("boom")
    ctx = OnNodeErrorContext(
        node_context=FakeNodeContext(),
        node_name="test_node",
        node_type="call_llm",
        error=original_err,
        state={},
    )
    result = await manager.run_hook("on_node_error", ctx)

    assert len(tracker.calls) == 1
    assert result.error is original_err


@pytest.mark.asyncio
async def test_on_node_error_consume() -> None:
    """When a hook sets error=None, the error is consumed."""
    manager = HookManager()
    manager.register(_ConsumeErrorExecutor())

    ctx = OnNodeErrorContext(
        node_context=FakeNodeContext(),
        node_name="test_node",
        node_type="call_llm",
        error=RuntimeError("boom"),
        state={},
    )
    result = await manager.run_hook("on_node_error", ctx)
    assert result.error is None


@pytest.mark.asyncio
async def test_on_node_error_reraise() -> None:
    """When a hook leaves the error untouched, it propagates."""
    tracker = _TrackingExecutor()
    manager = HookManager()
    manager.register(tracker)

    original_err = RuntimeError("boom")
    ctx = OnNodeErrorContext(
        node_context=FakeNodeContext(),
        node_name="test_node",
        node_type="call_llm",
        error=original_err,
        state={},
    )
    result = await manager.run_hook("on_node_error", ctx)
    assert result.error is original_err


@pytest.mark.asyncio
async def test_on_node_error_replace() -> None:
    """Hook can replace the error with a different exception."""
    manager = HookManager()
    manager.register(_ReplaceErrorExecutor())

    ctx = OnNodeErrorContext(
        node_context=FakeNodeContext(),
        node_name="test_node",
        node_type="call_llm",
        error=RuntimeError("original"),
        state={},
    )
    result = await manager.run_hook("on_node_error", ctx)
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "replaced"


@pytest.mark.asyncio
async def test_on_node_error_chain() -> None:
    """Multiple hooks run in order; each sees the previous hook's decision."""
    tracker = _TrackingExecutor()
    replacer = _ReplaceErrorExecutor()
    manager = HookManager()
    # tracker runs first (passes through), then replacer replaces
    manager.register(tracker)
    manager.register(replacer)

    original_err = RuntimeError("original")
    ctx = OnNodeErrorContext(
        node_context=FakeNodeContext(),
        node_name="test_node",
        node_type="call_llm",
        error=original_err,
        state={},
    )
    result = await manager.run_hook("on_node_error", ctx)

    # Both hooks ran
    assert len(tracker.calls) == 1
    # Final result has the replaced error from the second hook
    assert isinstance(result.error, ValueError)


# ---------------------------------------------------------------------------
# _run_node_error_hook helper tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_node_error_hook_no_hooks_raises() -> None:
    """Without hooks, the exception re-raises directly."""
    from sherma.langgraph.declarative.nodes import _run_node_error_hook

    exc = RuntimeError("no hooks")
    with pytest.raises(RuntimeError, match="no hooks"):
        await _run_node_error_hook(None, FakeNodeContext(), {}, exc)


@pytest.mark.asyncio
async def test_run_node_error_hook_consume_returns_empty() -> None:
    """When hooks consume the error, returns empty dict (fallback result)."""
    from sherma.langgraph.declarative.nodes import _run_node_error_hook

    manager = HookManager()
    manager.register(_ConsumeErrorExecutor())

    node_def = FakeNodeDef()
    ctx = FakeNodeContext(node_def=node_def, hook_manager=manager)
    result = await _run_node_error_hook(manager, ctx, {}, RuntimeError("consumed"))
    assert result == {}


@pytest.mark.asyncio
async def test_run_node_error_hook_reraise() -> None:
    """When hooks don't consume, the error is re-raised."""
    from sherma.langgraph.declarative.nodes import _run_node_error_hook

    manager = HookManager()
    manager.register(_TrackingExecutor())

    node_def = FakeNodeDef()
    ctx = FakeNodeContext(node_def=node_def, hook_manager=manager)
    with pytest.raises(RuntimeError, match="still here"):
        await _run_node_error_hook(manager, ctx, {}, RuntimeError("still here"))


# ---------------------------------------------------------------------------
# on_error (graph-level) tests
# ---------------------------------------------------------------------------


class _ConsumeGraphErrorExecutor(BaseHookExecutor):
    async def on_error(self, ctx: OnErrorContext) -> OnErrorContext | None:
        ctx.error = None
        return ctx


class _TrackGraphErrorExecutor(BaseHookExecutor):
    def __init__(self) -> None:
        self.calls: list[OnErrorContext] = []

    async def on_error(self, ctx: OnErrorContext) -> OnErrorContext | None:
        self.calls.append(ctx)
        return None


@pytest.mark.asyncio
async def test_on_error_fires_on_graph_invoke_failure() -> None:
    """on_error hook receives the exception from graph.ainvoke failure."""
    tracker = _TrackGraphErrorExecutor()
    manager = HookManager()
    manager.register(tracker)

    original_err = RuntimeError("graph boom")
    ctx = OnErrorContext(
        agent_id="agent-1",
        thread_id="thread-1",
        config={},
        input={"messages": []},
        error=original_err,
    )
    result = await manager.run_hook("on_error", ctx)

    assert len(tracker.calls) == 1
    assert result.error is original_err


@pytest.mark.asyncio
async def test_on_error_consume() -> None:
    """When on_error hook sets error=None, the error is swallowed."""
    manager = HookManager()
    manager.register(_ConsumeGraphErrorExecutor())

    ctx = OnErrorContext(
        agent_id="agent-1",
        thread_id="thread-1",
        config={},
        input={"messages": []},
        error=RuntimeError("graph boom"),
    )
    result = await manager.run_hook("on_error", ctx)
    assert result.error is None


# ---------------------------------------------------------------------------
# A2A executor error handling test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a2a_executor_catches_error_and_calls_failed() -> None:
    """ShermaAgentExecutor catches agent errors and calls task_updater.failed()."""
    from sherma.a2a.executor import ShermaAgentExecutor

    # Create a mock agent that raises on send_message
    mock_agent = MagicMock()
    mock_agent.input_schema = None
    mock_agent.output_schema = None

    async def _failing_send(*_args: Any, **_kwargs: Any):
        raise RuntimeError("agent exploded")
        yield  # makes this an async generator

    mock_agent.send_message = _failing_send

    executor = ShermaAgentExecutor(mock_agent)

    # Build context
    from a2a.types import Message, Part, Role, TextPart

    msg = Message(
        message_id="msg-1",
        role=Role.user,
        parts=[Part(root=TextPart(text="hello"))],
        task_id="task-1",
        context_id="ctx-1",
    )

    mock_context = MagicMock()
    mock_context.current_task = None
    mock_context.message = msg

    mock_event_queue = MagicMock()

    # Patch TaskUpdater methods
    with (
        patch("sherma.a2a.executor.TaskUpdater") as MockTaskUpdater,
        patch("sherma.a2a.executor.new_task") as mock_new_task,
    ):
        mock_task = MagicMock()
        mock_task.id = "task-1"
        mock_task.context_id = "ctx-1"
        mock_new_task.return_value = mock_task

        mock_updater_instance = MagicMock()
        mock_updater_instance.start_work = AsyncMock()
        mock_updater_instance.failed = AsyncMock()
        MockTaskUpdater.return_value = mock_updater_instance

        await executor.execute(mock_context, mock_event_queue)

        mock_updater_instance.failed.assert_called_once()
        call_kwargs = mock_updater_instance.failed.call_args
        error_msg = call_kwargs.kwargs.get("message") or call_kwargs.args[0]
        # Verify the error message mentions the failure
        text_parts = [p.root.text for p in error_msg.parts if hasattr(p.root, "text")]
        assert any("agent exploded" in t for t in text_parts)
