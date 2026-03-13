"""RemoteHookExecutor: delegates hook execution to a JSON-RPC 2.0 server."""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

import httpx

from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.serialization import deserialize_into_context, serialize_context
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
    OnErrorContext,
    OnNodeErrorContext,
)

logger = logging.getLogger(__name__)


class RemoteHookExecutor(BaseHookExecutor):
    """Hook executor that delegates to a remote JSON-RPC 2.0 server.

    Each hook is sent as a JSON-RPC method call.  The ``on_chat_model_create``
    hook is a no-op because it requires returning a Python object.

    On any network or protocol error the executor logs a warning and returns
    ``None`` (pass-through) so the agent is never blocked by a failing hook
    server.
    """

    _UNSUPPORTED_HOOKS: ClassVar[set[str]] = {"on_chat_model_create"}

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self._url = url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._request_id = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _call_rpc(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Send a JSON-RPC 2.0 request and return the result, or ``None``."""
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._request_id,
        }
        try:
            client = await self._get_client()
            resp = await client.post(
                self._url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Remote hook '%s' request failed: %s", method, exc)
            return None

        if "error" in body:
            logger.warning("Remote hook '%s' returned error: %s", method, body["error"])
            return None

        result = body.get("result")
        if result is None:
            return None
        if not isinstance(result, dict):
            logger.warning(
                "Remote hook '%s' returned non-dict result: %s", method, type(result)
            )
            return None
        return result

    async def _execute_hook(self, hook_name: str, ctx: Any) -> Any:
        """Serialize context, call RPC, deserialize response."""
        if hook_name in self._UNSUPPORTED_HOOKS:
            return None
        params = serialize_context(ctx)
        result = await self._call_rpc(hook_name, params)
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

    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
        return await self._execute_hook("node_enter", ctx)

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
