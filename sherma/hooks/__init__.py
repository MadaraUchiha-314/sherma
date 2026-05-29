"""Hooks system for agent lifecycle events."""

from sherma.hooks.apps import HookFastAPIApplication, HookStarletteApplication
from sherma.hooks.executor import BaseHookExecutor, HookExecutor
from sherma.hooks.manager import HookManager
from sherma.hooks.mcp_executor import MCPHookExecutor, MCPHookTransport
from sherma.hooks.mcp_server import MCPHookServer
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
    NodeExecuteContext,
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
    "HookManager",
    "HookStarletteApplication",
    "HookType",
    "MCPHookExecutor",
    "MCPHookServer",
    "MCPHookTransport",
    "NodeEnterContext",
    "NodeExecuteContext",
    "NodeExitContext",
    "OnErrorContext",
    "OnNodeErrorContext",
    "RemoteHookExecutor",
]
