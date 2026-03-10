"""Example hook executors for observability and prompt modification."""

from __future__ import annotations

import time

from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.types import (
    AfterLLMCallContext,
    AfterToolCallContext,
    BeforeLLMCallContext,
    BeforeToolCallContext,
    NodeEnterContext,
    NodeExitContext,
)


class LoggingHook(BaseHookExecutor):
    """Logs every node entry/exit and LLM/tool call for observability."""

    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
        print(f"  [hook] >>> Entering node '{ctx.node_name}' (type={ctx.node_type})")
        ctx.state["__node_enter_time__"] = time.monotonic()
        return None

    async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
        enter_time = ctx.state.get("__node_enter_time__")
        elapsed = ""
        if isinstance(enter_time, float):
            elapsed = f" ({time.monotonic() - enter_time:.3f}s)"
        print(f"  [hook] <<< Exiting node '{ctx.node_name}'{elapsed}")
        return None

    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        print(
            f"  [hook] LLM call: {len(ctx.messages)} messages, "
            f"{len(ctx.tools)} tools, prompt={ctx.system_prompt[:60]}..."
        )
        return None

    async def after_llm_call(
        self, ctx: AfterLLMCallContext
    ) -> AfterLLMCallContext | None:
        content = getattr(ctx.response, "content", "")
        tool_calls = getattr(ctx.response, "tool_calls", [])
        print(
            f"  [hook] LLM response: {len(str(content))} chars, "
            f"{len(tool_calls)} tool calls"
        )
        return None

    async def before_tool_call(
        self, ctx: BeforeToolCallContext
    ) -> BeforeToolCallContext | None:
        names = [tc.get("name", "?") for tc in ctx.tool_calls]
        print(f"  [hook] Tool calls: {names}")
        return None

    async def after_tool_call(
        self, ctx: AfterToolCallContext
    ) -> AfterToolCallContext | None:
        msg_count = len(ctx.result.get("messages", []))
        print(f"  [hook] Tool results: {msg_count} messages returned")
        return None


class PromptGuardrailHook(BaseHookExecutor):
    """Appends a safety guardrail to every LLM system prompt."""

    GUARDRAIL = (
        "\n\nIMPORTANT: Always be helpful, accurate, and concise. "
        "Never fabricate data — if a tool returns an error, say so."
    )

    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        ctx.system_prompt += self.GUARDRAIL
        return ctx
