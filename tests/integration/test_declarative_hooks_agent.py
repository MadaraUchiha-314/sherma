"""Integration test: declarative agent with hooks.

Verifies hook executors fire during agent execution and that
PromptGuardrailHook modifies the system prompt.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.types import (
    AfterLLMCallContext,
    BeforeLLMCallContext,
    NodeEnterContext,
    NodeExitContext,
)
from sherma.langgraph.declarative.agent import DeclarativeAgent
from sherma.langgraph.declarative.loader import RegistryBundle
from tests.integration.conftest import FakeChatModel, collect_events, make_a2a_message

HOOKS_AGENT_YAML = """\
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
  hooks-weather-agent:
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


class RecordingHook(BaseHookExecutor):
    """Records all hook invocations for test assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
        self.calls.append(("node_enter", ctx.node_name))
        return None

    async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
        self.calls.append(("node_exit", ctx.node_name))
        return None

    async def before_llm_call(
        self, ctx: BeforeLLMCallContext
    ) -> BeforeLLMCallContext | None:
        self.calls.append(("before_llm_call", ctx.node_name))
        return None

    async def after_llm_call(
        self, ctx: AfterLLMCallContext
    ) -> AfterLLMCallContext | None:
        self.calls.append(("after_llm_call", ctx.node_name))
        return None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hooks_fire_during_execution():
    """Recording hook captures node_enter, before/after_llm_call, node_exit."""
    fake_model = FakeChatModel(
        responses=[
            AIMessage(content="It's sunny in Tokyo!"),
        ],
    )
    registries = RegistryBundle(chat_models={"test-llm": fake_model})
    recording = RecordingHook()

    agent = DeclarativeAgent(
        id="hooks-weather-agent",
        version="1.0.0",
        yaml_content=HOOKS_AGENT_YAML,
        hooks=[recording],
    )
    agent._registries = registries

    message = make_a2a_message("Weather in Tokyo?")
    await collect_events(agent, message)

    hook_names = [name for name, _ in recording.calls]
    assert "node_enter" in hook_names
    assert "before_llm_call" in hook_names
    assert "after_llm_call" in hook_names
    assert "node_exit" in hook_names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompt_guardrail_appends_text():
    """PromptGuardrailHook from examples.hooks modifies the system prompt."""
    from examples.hooks import PromptGuardrailHook

    fake_model = FakeChatModel(
        responses=[
            AIMessage(content="Weather info here."),
        ],
    )
    registries = RegistryBundle(chat_models={"test-llm": fake_model})
    guardrail = PromptGuardrailHook()

    agent = DeclarativeAgent(
        id="hooks-weather-agent",
        version="1.0.0",
        yaml_content=HOOKS_AGENT_YAML,
        hooks=[guardrail],
    )
    agent._registries = registries

    message = make_a2a_message("Check weather")
    await collect_events(agent, message)

    # The FakeChatModel records the messages it received.
    # The first message list should contain the modified system prompt.
    assert fake_model.call_count >= 1
    first_call_messages = fake_model.received_messages[0]
    system_messages = [m for m in first_call_messages if m.type == "system"]
    assert len(system_messages) >= 1
    system_content = system_messages[0].content
    assert "IMPORTANT: Always be helpful" in system_content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_constructor_hooks_run():
    """Hooks passed via constructor fire during execution."""
    fake_model = FakeChatModel(
        responses=[
            AIMessage(content="Done."),
        ],
    )
    registries = RegistryBundle(chat_models={"test-llm": fake_model})
    recording = RecordingHook()

    agent = DeclarativeAgent(
        id="hooks-weather-agent",
        version="1.0.0",
        yaml_content=HOOKS_AGENT_YAML,
        hooks=[recording],
    )
    agent._registries = registries

    message = make_a2a_message("Hello")
    await collect_events(agent, message)

    # At minimum, node_enter should have fired
    assert len(recording.calls) > 0
    assert any(name == "node_enter" for name, _ in recording.calls)
