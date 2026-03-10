"""HookManager: orchestrates multiple hook executors."""

from __future__ import annotations

from typing import TypeVar

from sherma.hooks.executor import HookExecutor
from sherma.hooks.types import (
    AfterAgentCallContext,
    AfterInterruptContext,
    AfterLLMCallContext,
    AfterSkillLoadContext,
    AfterToolCallContext,
    BeforeAgentCallContext,
    BeforeInterruptContext,
    BeforeLLMCallContext,
    BeforeSkillLoadContext,
    BeforeToolCallContext,
    NodeEnterContext,
    NodeExitContext,
)

_ContextT = TypeVar(
    "_ContextT",
    BeforeLLMCallContext,
    AfterLLMCallContext,
    BeforeToolCallContext,
    AfterToolCallContext,
    BeforeAgentCallContext,
    AfterAgentCallContext,
    BeforeSkillLoadContext,
    AfterSkillLoadContext,
    NodeEnterContext,
    NodeExitContext,
    BeforeInterruptContext,
    AfterInterruptContext,
)


class HookManager:
    """Manages a list of hook executors and runs hooks in registration order.

    When running a hook, each executor is called in order. If an executor
    returns ``None``, the context passes through unchanged. If it returns
    a value, that replaces the context for subsequent executors.
    """

    def __init__(self) -> None:
        self._executors: list[HookExecutor] = []

    def register(self, executor: HookExecutor) -> None:
        """Register a hook executor."""
        self._executors.append(executor)

    async def run_hook(self, hook_name: str, ctx: _ContextT) -> _ContextT:
        """Run a named hook through all registered executors.

        Returns the (possibly modified) context after all executors have run.
        """
        for executor in self._executors:
            method = getattr(executor, hook_name, None)
            if method is None:
                continue
            result = await method(ctx)
            if result is not None:
                ctx = result
        return ctx
