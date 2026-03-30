"""HookExecutor protocol and BaseHookExecutor default implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sherma.hooks.types import (
    AfterAgentCallContext,
    AfterGraphInvokeContext,
    AfterInterruptContext,
    AfterLLMCallContext,
    AfterSkillLoadContext,
    AfterSkillUnloadContext,
    AfterToolCallContext,
    BeforeAgentCallContext,
    BeforeInterruptContext,
    BeforeLLMCallContext,
    BeforeSkillLoadContext,
    BeforeSkillUnloadContext,
    BeforeToolCallContext,
    ChatModelCreateContext,
    GraphInvokeContext,
    NodeEnterContext,
    NodeExecuteContext,
    NodeExitContext,
    OnErrorContext,
    OnNodeErrorContext,
)


@runtime_checkable
class HookExecutor(Protocol):
    """Protocol for hook executors.

    Each method corresponds to a lifecycle hook point. Return ``None``
    to pass the context through unchanged, or return a modified context
    to replace it for subsequent executors.
    """

    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None: ...

    async def after_llm_call(
        self, ctx: AfterLLMCallContext
    ) -> AfterLLMCallContext | None: ...

    async def before_tool_call(
        self, ctx: BeforeToolCallContext
    ) -> BeforeToolCallContext | None: ...

    async def after_tool_call(
        self, ctx: AfterToolCallContext
    ) -> AfterToolCallContext | None: ...

    async def before_agent_call(
        self, ctx: BeforeAgentCallContext
    ) -> BeforeAgentCallContext | None: ...

    async def after_agent_call(
        self, ctx: AfterAgentCallContext
    ) -> AfterAgentCallContext | None: ...

    async def before_skill_load(
        self, ctx: BeforeSkillLoadContext
    ) -> BeforeSkillLoadContext | None: ...

    async def after_skill_load(
        self, ctx: AfterSkillLoadContext
    ) -> AfterSkillLoadContext | None: ...

    async def before_skill_unload(
        self, ctx: BeforeSkillUnloadContext
    ) -> BeforeSkillUnloadContext | None: ...

    async def after_skill_unload(
        self, ctx: AfterSkillUnloadContext
    ) -> AfterSkillUnloadContext | None: ...

    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None: ...

    async def node_execute(
        self, ctx: NodeExecuteContext
    ) -> NodeExecuteContext | None: ...

    async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None: ...

    async def before_interrupt(
        self, ctx: BeforeInterruptContext
    ) -> BeforeInterruptContext | None: ...

    async def after_interrupt(
        self, ctx: AfterInterruptContext
    ) -> AfterInterruptContext | None: ...

    async def on_chat_model_create(
        self, ctx: ChatModelCreateContext
    ) -> ChatModelCreateContext | None: ...

    async def before_graph_invoke(
        self, ctx: GraphInvokeContext
    ) -> GraphInvokeContext | None: ...

    async def after_graph_invoke(
        self, ctx: AfterGraphInvokeContext
    ) -> AfterGraphInvokeContext | None: ...

    async def on_node_error(
        self, ctx: OnNodeErrorContext
    ) -> OnNodeErrorContext | None: ...

    async def on_error(self, ctx: OnErrorContext) -> OnErrorContext | None: ...


class BaseHookExecutor:
    """Default hook executor with all methods returning ``None``.

    Subclass and override only the hooks you need.
    """

    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        return None

    async def after_llm_call(
        self, ctx: AfterLLMCallContext
    ) -> AfterLLMCallContext | None:
        return None

    async def before_tool_call(
        self, ctx: BeforeToolCallContext
    ) -> BeforeToolCallContext | None:
        return None

    async def after_tool_call(
        self, ctx: AfterToolCallContext
    ) -> AfterToolCallContext | None:
        return None

    async def before_agent_call(
        self, ctx: BeforeAgentCallContext
    ) -> BeforeAgentCallContext | None:
        return None

    async def after_agent_call(
        self, ctx: AfterAgentCallContext
    ) -> AfterAgentCallContext | None:
        return None

    async def before_skill_load(
        self, ctx: BeforeSkillLoadContext
    ) -> BeforeSkillLoadContext | None:
        return None

    async def after_skill_load(
        self, ctx: AfterSkillLoadContext
    ) -> AfterSkillLoadContext | None:
        return None

    async def before_skill_unload(
        self, ctx: BeforeSkillUnloadContext
    ) -> BeforeSkillUnloadContext | None:
        return None

    async def after_skill_unload(
        self, ctx: AfterSkillUnloadContext
    ) -> AfterSkillUnloadContext | None:
        return None

    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
        return None

    async def node_execute(self, ctx: NodeExecuteContext) -> NodeExecuteContext | None:
        return None

    async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
        return None

    async def before_interrupt(
        self, ctx: BeforeInterruptContext
    ) -> BeforeInterruptContext | None:
        return None

    async def after_interrupt(
        self, ctx: AfterInterruptContext
    ) -> AfterInterruptContext | None:
        return None

    async def on_chat_model_create(
        self, ctx: ChatModelCreateContext
    ) -> ChatModelCreateContext | None:
        return None

    async def before_graph_invoke(
        self, ctx: GraphInvokeContext
    ) -> GraphInvokeContext | None:
        return None

    async def after_graph_invoke(
        self, ctx: AfterGraphInvokeContext
    ) -> AfterGraphInvokeContext | None:
        return None

    async def on_node_error(self, ctx: OnNodeErrorContext) -> OnNodeErrorContext | None:
        return None

    async def on_error(self, ctx: OnErrorContext) -> OnErrorContext | None:
        return None
