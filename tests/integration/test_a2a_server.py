"""Integration test: ShermaAgentExecutor processes A2A messages.

Tests the A2A executor wrapping a DeclarativeAgent with FakeChatModel.
"""

from __future__ import annotations

import asyncio

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import MessageSendParams, TaskState
from langchain_core.messages import AIMessage

from sherma.a2a.executor import ShermaAgentExecutor
from sherma.langgraph.declarative.agent import DeclarativeAgent
from sherma.langgraph.declarative.loader import RegistryBundle
from tests.integration.conftest import FakeChatModel, make_a2a_message

SIMPLE_AGENT_YAML = """\
prompts:
  - id: sys
    version: "1.0.0"
    instructions: "Be helpful."

llms:
  - id: test-llm
    version: "1.0.0"
    provider: openai
    model_name: test

agents:
  test-agent:
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
                content: 'prompts["sys"]["instructions"]'
              - role: messages
                content: 'messages'
      edges:
        - source: agent
          target: __end__
"""


async def _drain_event_queue(event_queue: EventQueue) -> list:
    """Drain all events from an EventQueue after execution completes."""
    events = []
    try:
        while True:
            event = event_queue.queue.get_nowait()
            events.append(event)
    except asyncio.QueueEmpty:
        pass
    return events


@pytest.mark.integration
@pytest.mark.asyncio
async def test_executor_processes_message():
    """ShermaAgentExecutor.execute sends events to EventQueue."""
    fake_model = FakeChatModel(
        responses=[AIMessage(content="Hello from the agent!")],
    )
    registries = RegistryBundle(chat_models={"test-llm": fake_model})

    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=SIMPLE_AGENT_YAML,
    )
    agent._registries = registries

    executor = ShermaAgentExecutor(agent=agent)
    event_queue = EventQueue()

    message = make_a2a_message("Hi there", task_id="task-1", context_id="ctx-1")
    request = MessageSendParams(message=message)
    context = RequestContext(request=request)

    await executor.execute(context, event_queue)

    events = await _drain_event_queue(event_queue)

    # Should have at least a working status and a completed status
    assert len(events) >= 1
    # The last event should be a completion
    last = events[-1]
    assert last.status.state in (TaskState.completed, TaskState.working)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_executor_handles_error_gracefully():
    """ShermaAgentExecutor handles agent errors and sends failed status."""
    fake_model = FakeChatModel(responses=[])  # Empty → will raise

    registries = RegistryBundle(chat_models={"test-llm": fake_model})

    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=SIMPLE_AGENT_YAML,
    )
    agent._registries = registries

    executor = ShermaAgentExecutor(agent=agent)
    event_queue = EventQueue()

    message = make_a2a_message("Hi", task_id="task-2", context_id="ctx-2")
    request = MessageSendParams(message=message)
    context = RequestContext(request=request)

    # Should not raise — executor catches and sends failed status
    await executor.execute(context, event_queue)

    events = await _drain_event_queue(event_queue)

    # Should have events including a failure
    assert len(events) >= 1
    assert any(e.status.state == TaskState.failed for e in events)
