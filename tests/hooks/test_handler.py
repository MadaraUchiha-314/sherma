"""Tests for HookHandler base class."""

from __future__ import annotations

from typing import Any

import pytest

from sherma.hooks.handler import HookHandler


@pytest.mark.asyncio
async def test_all_methods_return_none():
    """All default HookHandler methods return None."""
    handler = HookHandler()
    params: dict[str, Any] = {"node_name": "test"}

    assert await handler.before_llm_call(params) is None
    assert await handler.after_llm_call(params) is None
    assert await handler.before_tool_call(params) is None
    assert await handler.after_tool_call(params) is None
    assert await handler.before_agent_call(params) is None
    assert await handler.after_agent_call(params) is None
    assert await handler.before_skill_load(params) is None
    assert await handler.after_skill_load(params) is None
    assert await handler.node_enter(params) is None
    assert await handler.node_exit(params) is None
    assert await handler.before_interrupt(params) is None
    assert await handler.after_interrupt(params) is None
    assert await handler.before_graph_invoke(params) is None
    assert await handler.after_graph_invoke(params) is None
    assert await handler.on_node_error(params) is None
    assert await handler.on_error(params) is None


@pytest.mark.asyncio
async def test_subclass_override():
    """Subclassing HookHandler and overriding a method works."""

    class MyHooks(HookHandler):
        async def before_llm_call(
            self, params: dict[str, Any]
        ) -> dict[str, Any] | None:
            params["system_prompt"] = "modified"
            return params

    handler = MyHooks()
    params: dict[str, Any] = {"system_prompt": "original"}
    result = await handler.before_llm_call(params)
    assert result is not None
    assert result["system_prompt"] == "modified"

    # Other hooks still return None
    assert await handler.after_llm_call({}) is None


def test_no_on_chat_model_create():
    """HookHandler does not have on_chat_model_create."""
    assert not hasattr(HookHandler, "on_chat_model_create")
