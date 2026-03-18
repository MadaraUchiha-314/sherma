"""Integration test: declarative skill agent with skill discovery/execution.

Mirrors examples/declarative_skill_agent/agent.yaml with mocked LLM.
Tests the full skill loop: discover → execute → reflect.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from a2a.types import Message as A2AMessage
from langchain_core.messages import AIMessage

from sherma.langgraph.declarative.agent import DeclarativeAgent
from tests.integration.conftest import FakeChatModel, collect_events, make_a2a_message

# Use the real skill card from examples
SKILL_CARD_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "skills"
    / "weather"
    / "skill-card.json"
)

SKILL_AGENT_YAML = """\
manifest_version: 1

prompts:
  - id: discover-skills
    version: "1.0.0"
    instructions: >
      You have access to a catalog of skills. Given the user's request:
      1. Call list_skills ONCE to see available skills.
      2. Call load_skill_md ONCE for the most relevant skill.
      3. After loading, respond with a brief text summary.

  - id: plan-and-execute
    version: "1.0.0"
    instructions: >
      Based on the loaded skills, plan and execute the user's request.
      Use the available tools to accomplish the task. Be direct.

  - id: reflect
    version: "1.0.0"
    instructions: >
      Review the results so far. If you have enough information to give the
      user a useful answer, respond with "TASK_COMPLETE" followed by the answer.

  - id: summarize
    version: "1.0.0"
    instructions: >
      The task is not yet complete. Briefly summarize progress.

llms:
  - id: test-llm
    version: "1.0.0"
    provider: openai
    model_name: test

skills:
  - id: weather
    version: "1.0.0"
    skill_card_path: {skill_card_path}

agents:
  skill-agent:
    langgraph_config:
      recursion_limit: 50
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: discover_skills

      nodes:
        - name: discover_skills
          type: call_llm
          args:
            llm:
              id: test-llm
              version: "1.0.0"
            prompt:
              - role: system
                content: 'prompts["discover-skills"]["instructions"]'
              - role: messages
                content: 'messages'
            tools:
              - id: list_skills
              - id: load_skill_md

        - name: execute
          type: call_llm
          args:
            llm:
              id: test-llm
              version: "1.0.0"
            prompt:
              - role: system
                content: 'prompts["plan-and-execute"]["instructions"]'
              - role: messages
                content: 'messages'
            use_tools_from_loaded_skills: true

        - name: reflect
          type: call_llm
          args:
            llm:
              id: test-llm
              version: "1.0.0"
            prompt:
              - role: system
                content: 'prompts["reflect"]["instructions"]'
              - role: messages
                content: 'messages'

        - name: summarize
          type: call_llm
          args:
            llm:
              id: test-llm
              version: "1.0.0"
            prompt:
              - role: system
                content: 'prompts["summarize"]["instructions"]'
              - role: messages
                content: 'messages'

      edges:
        - source: discover_skills
          target: execute

        - source: execute
          target: reflect

        - source: reflect
          branches:
            - condition: >-
                messages[size(messages) - 1]["content"].contains("TASK_COMPLETE")
              target: __end__
          default: summarize

        - source: summarize
          target: execute
"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skill_discovery_and_execution(monkeypatch):
    """Full skill loop: discover → list_skills → load_skill_md → execute → reflect."""
    if not SKILL_CARD_PATH.exists():
        pytest.skip(f"Skill card not found at {SKILL_CARD_PATH}")

    yaml_content = SKILL_AGENT_YAML.replace("{skill_card_path}", str(SKILL_CARD_PATH))

    fake_model = FakeChatModel(
        responses=[
            # 1. discover_skills: call list_skills
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "list_skills",
                        "args": {},
                    }
                ],
            ),
            # 2. discover_skills (after tool result): call load_skill_md
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_2",
                        "name": "load_skill_md",
                        "args": {"skill_id": "weather", "version": "1.0.0"},
                    }
                ],
            ),
            # 3. discover_skills: text summary (no more tool calls)
            AIMessage(content="Found weather skill. Ready to execute."),
            # 4. execute: call get_weather tool
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_3",
                        "name": "get_weather",
                        "args": {"city": "Tokyo"},
                    }
                ],
            ),
            # 5. execute: text result after tool
            AIMessage(content="Tokyo weather: 20C, clear skies."),
            # 6. reflect: TASK_COMPLETE
            AIMessage(content="TASK_COMPLETE: Tokyo is 20C with clear skies."),
        ],
    )

    monkeypatch.setattr(
        "sherma.langgraph.declarative.loader._construct_chat_model",
        lambda provider, kwargs: fake_model,
    )

    agent = DeclarativeAgent(
        id="skill-agent",
        version="1.0.0",
        yaml_content=yaml_content,
        base_path=(
            Path(__file__).resolve().parent.parent.parent
            / "examples"
            / "declarative_skill_agent"
        ),
    )

    message = make_a2a_message("What's the weather in Tokyo?")
    events = await collect_events(agent, message)

    # Should complete with an A2A Message
    assert len(events) >= 1
    last = events[-1]
    assert isinstance(last, A2AMessage)
    text = last.parts[0].root.text
    assert "TASK_COMPLETE" in text or "Tokyo" in text
