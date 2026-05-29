"""Integration test: MCP-based hook server + executor end to end.

Spawns the example MCP hook server over stdio, connects a
``MCPHookExecutor`` to it, and verifies that lifecycle hooks
fire and that prompt modifications are visible in the chat model
call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from sherma.hooks.mcp_executor import MCPHookExecutor, MCPHookTransport
from sherma.hooks.types import BeforeLLMCallContext, NodeEnterContext
from sherma.langgraph.declarative.agent import DeclarativeAgent
from sherma.langgraph.declarative.loader import RegistryBundle
from tests.integration.conftest import FakeChatModel, collect_events, make_a2a_message

MCP_HOOK_AGENT_YAML = """\
manifest_version: 1

prompts:
  - id: weather-system-prompt
    version: "1.0.0"
    instructions: >
      You are a helpful weather assistant.

llms:
  - id: test-llm
    version: "1.0.0"
    provider: openai
    model_name: test

tools:
  - id: get_weather
    version: "1.0.0"
    import_path: tests.integration.fake_tools.get_weather

agents:
  mcp-hooks-weather-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: agent

      nodes:
        - name: agent
          type: call_llm
          args:
            llm:
              id: test-llm
              version: "1.0.0"
            prompt:
              - role: system
                content: 'prompts["weather-system-prompt"]["instructions"]'
              - role: messages
                content: 'state.messages'
            tools:
              - id: get_weather
                version: "1.0.0"
            state_updates:
              messages: '[llm_response]'

      edges:
        - source: agent
          target: __end__
"""


def _server_script_path() -> Path:
    """Return path to the example stdio MCP hook server."""
    return (
        Path(__file__).resolve().parent.parent.parent
        / "examples"
        / "mcp_hook_server"
        / "stdio_server.py"
    )


# ---------------------------------------------------------------------------
# Direct executor tests (no full agent)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_executor_discovers_overridden_hooks():
    """connect() populates available_hooks with the server's tools."""
    script = _server_script_path()
    assert script.exists(), f"Example script not found: {script}"

    transport = MCPHookTransport(command=sys.executable, args=[str(script)])
    executor = MCPHookExecutor(transport)
    try:
        await executor.connect()
        # The example server overrides before_llm_call, node_enter, node_exit,
        # and before_graph_invoke.
        assert "before_llm_call" in executor.available_hooks
        assert "node_enter" in executor.available_hooks
        assert "node_exit" in executor.available_hooks
        assert "before_graph_invoke" in executor.available_hooks
        # Hooks the example doesn't override should NOT be present
        assert "after_tool_call" not in executor.available_hooks
    finally:
        await executor.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_executor_before_llm_call_modifies_prompt():
    """The example server appends a guardrail to the system prompt."""
    script = _server_script_path()
    transport = MCPHookTransport(command=sys.executable, args=[str(script)])
    executor = MCPHookExecutor(transport)
    try:
        await executor.connect()
        ctx = BeforeLLMCallContext(
            node_context=None,
            node_name="agent",
            messages=[],
            system_prompt="You are a weather assistant.",
            tools=[],
            state={},
        )
        result = await executor.before_llm_call(ctx)
        assert result is not None
        assert "ALWAYS be helpful" in result.system_prompt
    finally:
        await executor.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_executor_node_enter_is_passthrough():
    """Logging hooks return None → executor returns None."""
    script = _server_script_path()
    transport = MCPHookTransport(command=sys.executable, args=[str(script)])
    executor = MCPHookExecutor(transport)
    try:
        await executor.connect()
        ctx = NodeEnterContext(
            node_context=None,
            node_name="agent",
            node_type="call_llm",
            state={},
        )
        result = await executor.node_enter(ctx)
        assert result is None
    finally:
        await executor.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_executor_skipped_hook_returns_none_without_call():
    """Hooks not registered by the server are skipped (no IPC, no error)."""
    script = _server_script_path()
    transport = MCPHookTransport(command=sys.executable, args=[str(script)])
    executor = MCPHookExecutor(transport)
    try:
        await executor.connect()
        # The example doesn't override after_tool_call
        from sherma.hooks.types import AfterToolCallContext

        ctx = AfterToolCallContext(
            node_context=None,
            node_name="x",
            result={"ok": True},
            state={},
        )
        result = await executor.after_tool_call(ctx)
        assert result is None
    finally:
        await executor.close()


# ---------------------------------------------------------------------------
# Full agent integration: hook server modifies the system prompt the LLM sees
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_hook_modifies_system_prompt_in_agent():
    """End-to-end: the MCP-hook-modified system prompt reaches the chat model."""
    script = _server_script_path()
    transport = MCPHookTransport(command=sys.executable, args=[str(script)])
    executor = MCPHookExecutor(transport)
    await executor.connect()
    try:
        fake_model = FakeChatModel(
            responses=[AIMessage(content="It's sunny in Tokyo!")]
        )
        registries = RegistryBundle(chat_models={"test-llm": fake_model})

        agent = DeclarativeAgent(
            id="mcp-hooks-weather-agent",
            version="1.0.0",
            yaml_content=MCP_HOOK_AGENT_YAML,
            hooks=[executor],
        )
        agent._registries = registries

        message = make_a2a_message("Weather in Tokyo?")
        await collect_events(agent, message)

        assert fake_model.call_count >= 1
        first_call_messages = fake_model.received_messages[0]
        system_messages = [m for m in first_call_messages if m.type == "system"]
        assert len(system_messages) >= 1
        # Hook injected its guardrail
        assert "ALWAYS be helpful" in system_messages[0].content
    finally:
        await executor.close()


# ---------------------------------------------------------------------------
# YAML-driven loader test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_yaml_mcp_hook_definition_connects_and_modifies_prompt():
    """A YAML-declared MCP hook is connected by the loader and fires."""
    script = _server_script_path()
    yaml_with_mcp_hook = f"""\
manifest_version: 1

prompts:
  - id: weather-system-prompt
    version: "1.0.0"
    instructions: >
      You are a helpful weather assistant.

llms:
  - id: test-llm
    version: "1.0.0"
    provider: openai
    model_name: test

tools:
  - id: get_weather
    version: "1.0.0"
    import_path: tests.integration.fake_tools.get_weather

hooks:
  - mcp:
      command: {sys.executable!r}
      args: [{str(script)!r}]

agents:
  yaml-mcp-hooks-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: agent

      nodes:
        - name: agent
          type: call_llm
          args:
            llm:
              id: test-llm
              version: "1.0.0"
            prompt:
              - role: system
                content: 'prompts["weather-system-prompt"]["instructions"]'
              - role: messages
                content: 'state.messages'
            tools:
              - id: get_weather
                version: "1.0.0"
            state_updates:
              messages: '[llm_response]'

      edges:
        - source: agent
          target: __end__
"""
    fake_model = FakeChatModel(responses=[AIMessage(content="ok")])
    registries = RegistryBundle(chat_models={"test-llm": fake_model})

    agent = DeclarativeAgent(
        id="yaml-mcp-hooks-agent",
        version="1.0.0",
        yaml_content=yaml_with_mcp_hook,
    )
    agent._registries = registries

    try:
        message = make_a2a_message("Hi")
        await collect_events(agent, message)

        assert fake_model.call_count >= 1
        first_call_messages = fake_model.received_messages[0]
        system_messages = [m for m in first_call_messages if m.type == "system"]
        assert any("ALWAYS be helpful" in m.content for m in system_messages)
    finally:
        await agent.aclose()


# ---------------------------------------------------------------------------
# Server-side unit-style integration: call_tool directly on the MCPHookServer
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_hook_server_serves_overridden_hooks():
    """The MCPHookServer routes call_tool to the underlying executor."""
    from sherma.hooks.executor import BaseHookExecutor
    from sherma.hooks.mcp_server import HOOK_TOOL_PREFIX, MCPHookServer

    class MyHooks(BaseHookExecutor):
        async def before_llm_call(
            self, ctx: BeforeLLMCallContext
        ) -> BeforeLLMCallContext | None:
            ctx.system_prompt = (ctx.system_prompt or "") + " [hooked]"
            return ctx

        async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
            return None

    server = MCPHookServer(executor=MyHooks(), name="test-srv")
    assert server.registered_hooks == {"before_llm_call", "node_enter"}

    tool_name = f"{HOOK_TOOL_PREFIX}before_llm_call"
    result = await server.server.call_tool(
        tool_name, {"params": {"system_prompt": "orig", "node_name": "n", "state": {}}}
    )
    if isinstance(result, tuple):
        _, structured = result
    else:
        structured = None
    assert structured is not None
    inner = structured.get("result", structured)
    assert inner["system_prompt"] == "orig [hooked]"
