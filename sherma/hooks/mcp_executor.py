"""MCPHookExecutor: delegates hook execution to an MCP server.

An MCP hook server exposes one tool per hook type it handles, named
``hooks.<hook_name>`` (e.g. ``hooks.before_llm_call``).  The executor
discovers available hooks via ``list_tools`` at connection time and
only dispatches to those.

Slash characters are not allowed in MCP tool names (see SEP-986), so
the namespace separator is ``.`` rather than ``/``.

On any error the executor logs a warning and returns ``None``
(pass-through) so the agent is never blocked by a failing hook server.
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.serialization import deserialize_into_context, serialize_context
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
    GraphInvokeContext,
    NodeEnterContext,
    NodeExecuteContext,
    NodeExitContext,
    OnErrorContext,
    OnNodeErrorContext,
)

logger = logging.getLogger(__name__)

HOOK_TOOL_PREFIX = "hooks."


@dataclass
class MCPHookTransport:
    """Connection parameters for an MCP hook server.

    Provide exactly one of:

    * ``command`` (+ optional ``args`` / ``env``) for a stdio transport.
    * ``url`` for an HTTP transport (``streamable_http`` or ``sse``).
    """

    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    transport: Literal["streamable_http", "sse"] = "streamable_http"
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.command) == bool(self.url):
            raise ValueError(
                "MCPHookTransport requires exactly one of 'command' or 'url'"
            )


class MCPHookExecutor(BaseHookExecutor):
    """Hook executor that delegates to an MCP server.

    Each hook is invoked via the MCP ``call_tool`` method on a tool
    named ``hooks.<hook_name>``.  The ``on_chat_model_create`` hook is
    a no-op because it requires returning a Python object.

    Connection lifecycle is managed via :meth:`connect` and
    :meth:`close`.  After :meth:`connect`, the executor knows which
    hook tools the server provides and skips any hook not in that set.
    """

    _UNSUPPORTED_HOOKS: ClassVar[set[str]] = {"on_chat_model_create"}

    def __init__(self, transport: MCPHookTransport) -> None:
        self._transport = transport
        self._exit_stack: AsyncExitStack | None = None
        self._session: Any = None  # mcp.ClientSession after connect
        self._available_hooks: set[str] = set()

    @property
    def available_hooks(self) -> set[str]:
        """Hook names the MCP server has registered as tools."""
        return set(self._available_hooks)

    async def connect(self) -> None:
        """Open the MCP connection and discover available hook tools."""
        if self._session is not None:
            return

        from mcp import ClientSession

        stack = AsyncExitStack()
        try:
            if self._transport.command is not None:
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                params = StdioServerParameters(
                    command=self._transport.command,
                    args=list(self._transport.args),
                    env=dict(self._transport.env) if self._transport.env else None,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
            elif self._transport.transport == "sse":
                from mcp.client.sse import sse_client

                read, write = await stack.enter_async_context(
                    sse_client(
                        url=self._transport.url,  # type: ignore[arg-type]
                        headers=dict(self._transport.headers)
                        if self._transport.headers
                        else None,
                    )
                )
                session = await stack.enter_async_context(ClientSession(read, write))
            else:  # streamable_http
                from mcp.client.streamable_http import streamablehttp_client

                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(
                        url=self._transport.url,  # type: ignore[arg-type]
                        headers=dict(self._transport.headers)
                        if self._transport.headers
                        else None,
                    )
                )
                session = await stack.enter_async_context(ClientSession(read, write))

            await session.initialize()
            tools_result = await session.list_tools()
            self._available_hooks = {
                t.name.removeprefix(HOOK_TOOL_PREFIX)
                for t in tools_result.tools
                if t.name.startswith(HOOK_TOOL_PREFIX)
            }
            self._session = session
            self._exit_stack = stack
        except Exception:
            await stack.aclose()
            raise

    async def close(self) -> None:
        """Close the MCP connection if open."""
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Error closing MCP hook connection: %s", exc)
        self._exit_stack = None
        self._session = None
        self._available_hooks = set()

    async def _call_tool(
        self, hook_name: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Call the MCP tool for *hook_name* and parse the result."""
        if self._session is None:
            logger.warning(
                "MCP hook '%s' called before connect(); returning pass-through",
                hook_name,
            )
            return None

        tool_name = f"{HOOK_TOOL_PREFIX}{hook_name}"
        try:
            result = await self._session.call_tool(tool_name, {"params": params})
        except Exception as exc:
            logger.warning("MCP hook '%s' call failed: %s", hook_name, exc)
            return None

        if result.isError:
            logger.warning(
                "MCP hook '%s' returned error: %s",
                hook_name,
                _stringify_content(result.content),
            )
            return None

        # Prefer structured content when available (FastMCP wraps a
        # ``dict | None`` return as ``{"result": ...}`` in structuredContent).
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            value = (
                structured.get("result", structured)
                if isinstance(structured, dict)
                else structured
            )
            if value is None:
                return None
            if isinstance(value, dict):
                return value
            logger.warning(
                "MCP hook '%s' returned non-dict structured result: %s",
                hook_name,
                type(value).__name__,
            )
            return None

        text = _stringify_content(result.content)
        if not text or text == "null":
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "MCP hook '%s' returned non-JSON content: %s", hook_name, exc
            )
            return None
        if parsed is None:
            return None
        if not isinstance(parsed, dict):
            logger.warning(
                "MCP hook '%s' returned non-dict content: %s",
                hook_name,
                type(parsed).__name__,
            )
            return None
        return parsed

    async def _execute_hook(self, hook_name: str, ctx: Any) -> Any:
        if hook_name in self._UNSUPPORTED_HOOKS:
            return None
        if hook_name not in self._available_hooks:
            # Server did not register this hook → skip without any RPC
            return None
        params = serialize_context(ctx)
        # MCP transports require strictly JSON-safe params. ``_serialize_complex``
        # in the shared serializer may leave non-JSON values (class objects,
        # functions) inside dumped LangChain models, so coerce one more time.
        params = _json_safe(params)
        result = await self._call_tool(hook_name, params)
        if result is None:
            return None
        return deserialize_into_context(type(ctx), result, ctx)

    # -- Hook method overrides --

    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        return await self._execute_hook("before_llm_call", ctx)

    async def after_llm_call(
        self, ctx: AfterLLMCallContext
    ) -> AfterLLMCallContext | None:
        return await self._execute_hook("after_llm_call", ctx)

    async def before_tool_call(
        self, ctx: BeforeToolCallContext
    ) -> BeforeToolCallContext | None:
        return await self._execute_hook("before_tool_call", ctx)

    async def after_tool_call(
        self, ctx: AfterToolCallContext
    ) -> AfterToolCallContext | None:
        return await self._execute_hook("after_tool_call", ctx)

    async def before_agent_call(
        self, ctx: BeforeAgentCallContext
    ) -> BeforeAgentCallContext | None:
        return await self._execute_hook("before_agent_call", ctx)

    async def after_agent_call(
        self, ctx: AfterAgentCallContext
    ) -> AfterAgentCallContext | None:
        return await self._execute_hook("after_agent_call", ctx)

    async def before_skill_load(
        self, ctx: BeforeSkillLoadContext
    ) -> BeforeSkillLoadContext | None:
        return await self._execute_hook("before_skill_load", ctx)

    async def after_skill_load(
        self, ctx: AfterSkillLoadContext
    ) -> AfterSkillLoadContext | None:
        return await self._execute_hook("after_skill_load", ctx)

    async def before_skill_unload(
        self, ctx: BeforeSkillUnloadContext
    ) -> BeforeSkillUnloadContext | None:
        return await self._execute_hook("before_skill_unload", ctx)

    async def after_skill_unload(
        self, ctx: AfterSkillUnloadContext
    ) -> AfterSkillUnloadContext | None:
        return await self._execute_hook("after_skill_unload", ctx)

    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
        return await self._execute_hook("node_enter", ctx)

    async def node_execute(self, ctx: NodeExecuteContext) -> NodeExecuteContext | None:
        return await self._execute_hook("node_execute", ctx)

    async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
        return await self._execute_hook("node_exit", ctx)

    async def before_interrupt(
        self, ctx: BeforeInterruptContext
    ) -> BeforeInterruptContext | None:
        return await self._execute_hook("before_interrupt", ctx)

    async def after_interrupt(
        self, ctx: AfterInterruptContext
    ) -> AfterInterruptContext | None:
        return await self._execute_hook("after_interrupt", ctx)

    # on_chat_model_create: inherits no-op from BaseHookExecutor

    async def before_graph_invoke(
        self, ctx: GraphInvokeContext
    ) -> GraphInvokeContext | None:
        return await self._execute_hook("before_graph_invoke", ctx)

    async def after_graph_invoke(
        self, ctx: AfterGraphInvokeContext
    ) -> AfterGraphInvokeContext | None:
        return await self._execute_hook("after_graph_invoke", ctx)

    async def on_node_error(self, ctx: OnNodeErrorContext) -> OnNodeErrorContext | None:
        return await self._execute_hook("on_node_error", ctx)

    async def on_error(self, ctx: OnErrorContext) -> OnErrorContext | None:
        return await self._execute_hook("on_error", ctx)


def _stringify_content(content: list[Any]) -> str:
    """Concatenate the text portions of an MCP CallToolResult.content list."""
    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    return "".join(parts)


def _json_safe(value: Any) -> Any:
    """Recursively coerce *value* to a strictly JSON-serializable form.

    Primitives pass through.  Dicts and lists are walked.  Anything else
    is converted via ``repr()``.  This is best-effort and lossy for
    complex LangChain objects — but those are read-only hints for hook
    servers anyway.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)
