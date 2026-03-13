"""Hooks system for agent lifecycle events."""

from sherma.hooks.apps import HookFastAPIApplication, HookStarletteApplication
from sherma.hooks.executor import BaseHookExecutor, HookExecutor
from sherma.hooks.handler import HookHandler
from sherma.hooks.manager import HookManager
from sherma.hooks.remote import RemoteHookExecutor
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
    ChatModelCreateContext,
    GraphInvokeContext,
    HookType,
    NodeEnterContext,
    NodeExitContext,
    OnErrorContext,
    OnNodeErrorContext,
)

__all__ = [
    "AfterAgentCallContext",
    "AfterGraphInvokeContext",
    "AfterInterruptContext",
    "AfterLLMCallContext",
    "AfterSkillLoadContext",
    "AfterToolCallContext",
    "BaseHookExecutor",
    "BeforeAgentCallContext",
    "BeforeInterruptContext",
    "BeforeLLMCallContext",
    "BeforeSkillLoadContext",
    "BeforeToolCallContext",
    "ChatModelCreateContext",
    "GraphInvokeContext",
    "HookExecutor",
    "HookFastAPIApplication",
    "HookHandler",
    "HookManager",
    "HookStarletteApplication",
    "HookType",
    "NodeEnterContext",
    "NodeExitContext",
    "OnErrorContext",
    "OnNodeErrorContext",
    "RemoteHookExecutor",
]
