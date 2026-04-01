"""Integration test: declarative error recovery with retry and fallback.

Tests the full DeclarativeAgent lifecycle with on_error configuration.
"""

from __future__ import annotations

import pytest
from a2a.types import Message
from langchain_core.messages import AIMessage

from sherma.langgraph.declarative.agent import DeclarativeAgent
from sherma.langgraph.declarative.loader import RegistryBundle
from tests.integration.conftest import (
    FailingChatModel,
    FakeChatModel,
    collect_events,
    make_a2a_message,
)

# ---- YAML: call_llm with retry + fallback ----

RETRY_FALLBACK_YAML = """\
manifest_version: 1

llms:
  - id: test-llm
    version: "1.0.0"
    provider: openai
    model_name: test

agents:
  retry-agent:
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
            llm: { id: test-llm }
            prompt:
              - role: system
                content: '"You are helpful."'
              - role: messages
                content: state.messages
            state_updates:
              messages: '[llm_response]'
          on_error:
            retry:
              max_attempts: 3
              strategy: fixed
              delay: 0.01
            fallback: error_handler

        - name: error_handler
          type: data_transform
          args:
            expression: >
              {
                "messages": [
                  "Error: "
                  + state["__sherma__"]["last_error"]["message"]
                ]
              }

      edges:
        - source: agent
          target: __end__
        - source: error_handler
          target: __end__
"""


# ---- YAML: call_llm with retry only (no fallback) ----

RETRY_ONLY_YAML = """\
manifest_version: 1

llms:
  - id: test-llm
    version: "1.0.0"
    provider: openai
    model_name: test

agents:
  retry-only-agent:
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
            llm: { id: test-llm }
            prompt:
              - role: system
                content: '"You are helpful."'
              - role: messages
                content: state.messages
            state_updates:
              messages: '[llm_response]'
          on_error:
            retry:
              max_attempts: 3
              strategy: fixed
              delay: 0.01

      edges:
        - source: agent
          target: __end__
"""


# ---- YAML: tool_node with fallback (no retry) ----

TOOL_FALLBACK_YAML = """\
manifest_version: 1

llms:
  - id: test-llm
    version: "1.0.0"
    provider: openai
    model_name: test

tools:
  - id: failing_tool
    version: "1.0.0"
    import_path: tests.integration.fake_tools.get_weather

agents:
  tool-fallback-agent:
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
            llm: { id: test-llm }
            prompt:
              - role: system
                content: '"You are helpful."'
              - role: messages
                content: state.messages
            tools:
              - id: failing_tool
            state_updates:
              messages: '[llm_response]'

        - name: agent_tools
          type: tool_node
          args: {}
          on_error:
            fallback: tool_error_handler

        - name: tool_error_handler
          type: data_transform
          args:
            expression: >
              {
                "messages": [
                  "Tool failed: "
                  + state["__sherma__"]["last_error"]["message"]
                ]
              }

      edges:
        - source: agent
          branches:
            - condition: has_tool_calls
              target: agent_tools
          default: __end__

        - source: agent_tools
          target: agent

        - source: tool_error_handler
          target: __end__
"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failures():
    """LLM fails twice then succeeds on 3rd attempt (within retry budget)."""
    model = FailingChatModel(
        fail_count=2,
        error=RuntimeError("rate limit exceeded"),
        success_response=AIMessage(content="Hello! I recovered."),
    )
    registries = RegistryBundle(chat_models={"test-llm": model})
    agent = DeclarativeAgent(
        id="retry-agent",
        version="1.0.0",
        yaml_content=RETRY_FALLBACK_YAML,
    )
    agent._registries = registries

    msg = make_a2a_message("Hi")
    events = await collect_events(agent, msg)

    # Should get a successful response
    messages = [e for e in events if isinstance(e, Message)]
    assert len(messages) == 1
    text = messages[0].parts[0].root.text
    assert "recovered" in text.lower()

    # Model was called 3 times (2 failures + 1 success)
    assert model.call_count == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_exhausted_routes_to_fallback():
    """LLM fails all 3 attempts, error_handler node produces response."""
    model = FailingChatModel(
        fail_count=10,  # always fails
        error=RuntimeError("service unavailable"),
        success_response=AIMessage(content="never reached"),
    )
    registries = RegistryBundle(chat_models={"test-llm": model})
    agent = DeclarativeAgent(
        id="retry-agent",
        version="1.0.0",
        yaml_content=RETRY_FALLBACK_YAML,
    )
    agent._registries = registries

    msg = make_a2a_message("Hi")
    events = await collect_events(agent, msg)

    # Should get a response from error_handler
    messages = [e for e in events if isinstance(e, Message)]
    assert len(messages) == 1
    text = messages[0].parts[0].root.text
    assert "service unavailable" in text.lower()

    # Model was called exactly 3 times (max_attempts)
    assert model.call_count == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_exhausted_no_fallback_raises():
    """LLM fails all attempts, no fallback → exception propagates."""
    model = FailingChatModel(
        fail_count=10,
        error=RuntimeError("total failure"),
        success_response=AIMessage(content="never"),
    )
    registries = RegistryBundle(chat_models={"test-llm": model})
    agent = DeclarativeAgent(
        id="retry-only-agent",
        version="1.0.0",
        yaml_content=RETRY_ONLY_YAML,
    )
    agent._registries = registries

    msg = make_a2a_message("Hi")
    with pytest.raises(RuntimeError, match="total failure"):
        await collect_events(agent, msg)

    assert model.call_count == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_on_error_single_attempt():
    """Without on_error, LLM is called once and succeeds normally."""
    yaml = """\
manifest_version: 1

llms:
  - id: test-llm
    version: "1.0.0"
    provider: openai
    model_name: test

agents:
  simple-agent:
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
            llm: { id: test-llm }
            prompt:
              - role: system
                content: '"You are helpful."'
              - role: messages
                content: state.messages
            state_updates:
              messages: '[llm_response]'

      edges:
        - source: agent
          target: __end__
"""
    model = FakeChatModel(responses=[AIMessage(content="Hi there!")])
    registries = RegistryBundle(chat_models={"test-llm": model})
    agent = DeclarativeAgent(
        id="simple-agent",
        version="1.0.0",
        yaml_content=yaml,
    )
    agent._registries = registries

    msg = make_a2a_message("Hello")
    events = await collect_events(agent, msg)

    messages = [e for e in events if isinstance(e, Message)]
    assert len(messages) == 1
    assert "Hi there!" in messages[0].parts[0].root.text
    assert model.call_count == 1
