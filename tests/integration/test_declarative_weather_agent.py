"""Integration test: declarative weather agent with interrupt/resume.

Mirrors examples/declarative_weather_agent/agent.yaml with mocked LLM.
"""

from __future__ import annotations

import pytest
from a2a.types import TaskStatusUpdateEvent
from langchain_core.messages import AIMessage

from sherma.langgraph.declarative.agent import DeclarativeAgent
from sherma.langgraph.declarative.loader import RegistryBundle
from tests.integration.conftest import FakeChatModel, collect_events, make_a2a_message

WEATHER_AGENT_YAML = """\
prompts:
  - id: weather-system-prompt
    version: "1.0.0"
    instructions: >
      You are a helpful weather assistant.
      Use the get_weather tool to look up current weather.

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
  weather-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
        - name: query_count
          type: int
          default: 0
        - name: last_status
          type: str
          default: ""

    graph:
      entry_point: init

      nodes:
        - name: init
          type: set_state
          args:
            values:
              query_count: "0"
              last_status: '"ready"'

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
                content: 'messages'
            tools:
              - id: get_weather
                version: "1.0.0"

        - name: update_stats
          type: data_transform
          args:
            expression: '{"query_count": query_count + 1, "last_status": "done"}'

        - name: ask_next_place
          type: interrupt
          args: {}

      edges:
        - source: init
          target: agent

        - source: agent
          target: update_stats

        - source: update_stats
          target: ask_next_place

        - source: ask_next_place
          target: agent
"""


def _make_agent(responses: list[AIMessage]) -> DeclarativeAgent:
    fake_model = FakeChatModel(responses=responses)
    registries = RegistryBundle(chat_models={"test-llm": fake_model})
    agent = DeclarativeAgent(
        id="weather-agent",
        version="1.0.0",
        yaml_content=WEATHER_AGENT_YAML,
    )
    agent._registries = registries
    return agent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basic_query_with_interrupt():
    """First send_message goes through init → agent → update_stats → interrupt."""
    agent = _make_agent(
        responses=[
            # LLM calls get_weather tool
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                    }
                ],
            ),
            # LLM final answer
            AIMessage(content="Paris is 20C with clear skies."),
        ],
    )

    message = make_a2a_message("What is the weather in Paris?")
    events = await collect_events(agent, message)

    # Should yield a TaskStatusUpdateEvent with input_required (from interrupt)
    status_events = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
    assert len(status_events) >= 1
    assert status_events[0].status.state.value == "input-required"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interrupt_resume():
    """After interrupt, send follow-up message to resume the agent."""
    fake_model = FakeChatModel(
        responses=[
            # First turn: tool call
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                    }
                ],
            ),
            # First turn: final answer
            AIMessage(content="Paris is 20C with clear skies."),
            # Second turn after resume: tool call
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_2",
                        "name": "get_weather",
                        "args": {"city": "London"},
                    }
                ],
            ),
            # Second turn: final answer
            AIMessage(content="London is 15C with light rain."),
        ],
    )
    registries = RegistryBundle(chat_models={"test-llm": fake_model})
    agent = DeclarativeAgent(
        id="weather-agent",
        version="1.0.0",
        yaml_content=WEATHER_AGENT_YAML,
    )
    agent._registries = registries

    # First message — hits interrupt
    msg1 = make_a2a_message(
        "What is the weather in Paris?",
        context_id="ctx-1",
    )
    events1 = await collect_events(agent, msg1)
    status_events = [e for e in events1 if isinstance(e, TaskStatusUpdateEvent)]
    assert len(status_events) >= 1

    # Resume with second message (same context_id triggers resume)
    msg2 = make_a2a_message(
        "Now check London.",
        message_id="user-2",
        context_id="ctx-1",
    )
    events2 = await collect_events(agent, msg2)

    # Second turn should also interrupt (the graph loops back)
    status_events2 = [e for e in events2 if isinstance(e, TaskStatusUpdateEvent)]
    assert len(status_events2) >= 1

    # Model should have been called 4 times total
    assert fake_model.call_count == 4
