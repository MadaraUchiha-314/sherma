"""Integration test: multi-agent supervisor with sub-agent delegation.

Mirrors examples/multi_agent/ with monkeypatched LLM construction.
"""

from __future__ import annotations

import pytest
from a2a.types import Message as A2AMessage
from langchain_core.messages import AIMessage

from sherma.langgraph.declarative.agent import DeclarativeAgent
from tests.integration.conftest import FakeChatModel, collect_events, make_a2a_message

# Inline the weather sub-agent YAML (uses fake tool)
WEATHER_AGENT_YAML = """\
prompts:
  - id: weather-prompt
    version: "1.0.0"
    instructions: >
      You are a weather assistant.
      Use the get_weather tool to look up current weather.
      Return a concise weather summary.

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
                content: 'prompts["weather-prompt"]["instructions"]'
              - role: messages
                content: 'messages'
            tools:
              - id: get_weather
                version: "1.0.0"
      edges:
        - source: agent
          target: __end__
"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_supervisor_delegates_to_sub_agent(tmp_path, monkeypatch):
    """Supervisor calls weather-agent tool, which calls get_weather, full chain."""
    # Write the weather sub-agent YAML to tmp_path
    weather_yaml_path = tmp_path / "weather_agent.yaml"
    weather_yaml_path.write_text(WEATHER_AGENT_YAML)

    # Build supervisor YAML referencing the sub-agent
    supervisor_yaml = f"""\
prompts:
  - id: supervisor-prompt
    version: "1.0.0"
    instructions: >
      You are a travel planner. Use the weather-agent to check weather
      and give travel advice.

llms:
  - id: test-llm
    version: "1.0.0"
    provider: openai
    model_name: test

sub_agents:
  - id: weather-agent
    version: "1.0.0"
    yaml_path: {weather_yaml_path}

agents:
  travel-planner:
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: planner
      nodes:
        - name: planner
          type: call_llm
          args:
            llm:
              id: test-llm
              version: "1.0.0"
            prompt:
              - role: system
                content: 'prompts["supervisor-prompt"]["instructions"]'
              - role: messages
                content: 'messages'
            use_sub_agents_as_tools: true

      edges:
        - source: planner
          target: __end__
"""

    # Sub-agent responses: tool call → text answer
    sub_agent_responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_sub_1",
                    "name": "get_weather",
                    "args": {"city": "Tokyo"},
                }
            ],
        ),
        AIMessage(content="Tokyo is 20C, clear skies."),
    ]

    # Supervisor responses: call weather-agent tool → final text
    supervisor_responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_sup_1",
                    "name": "weather-agent",
                    "args": {"request": "What is the weather in Tokyo?"},
                }
            ],
        ),
        AIMessage(content="Tokyo has great weather at 20C! Pack light clothes."),
    ]

    # Monkeypatch _construct_chat_model so both supervisor and sub-agent
    # get FakeChatModel instances
    models: dict[str, FakeChatModel] = {}

    def fake_construct(provider: str, kwargs: dict) -> FakeChatModel:
        # Each unique provider+model call gets its own FakeChatModel
        # The supervisor is created first, then the sub-agent
        if len(models) == 0:
            model = FakeChatModel(responses=supervisor_responses)
            models["supervisor"] = model
            return model
        else:
            model = FakeChatModel(responses=sub_agent_responses)
            models["sub_agent"] = model
            return model

    monkeypatch.setattr(
        "sherma.langgraph.declarative.loader._construct_chat_model",
        fake_construct,
    )

    agent = DeclarativeAgent(
        id="travel-planner",
        version="1.0.0",
        yaml_content=supervisor_yaml,
    )

    message = make_a2a_message("Plan a trip to Tokyo")
    events = await collect_events(agent, message)

    # Should get an A2A Message response
    assert len(events) >= 1
    last = events[-1]
    assert isinstance(last, A2AMessage)
    text = last.parts[0].root.text
    assert "Tokyo" in text or "20C" in text
