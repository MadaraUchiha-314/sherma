"""Tests for DeclarativeAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from sherma.langgraph.declarative.agent import DeclarativeAgent
from sherma.langgraph.declarative.loader import RegistryBundle

SET_STATE_YAML = """\
agents:
  test-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
        - name: result
          type: str
          default: ""
    graph:
      entry_point: setter
      nodes:
        - name: setter
          type: set_state
          args:
            values:
              result: '"done"'
      edges: []
"""

DATA_TRANSFORM_YAML = """\
agents:
  test-agent:
    state:
      fields:
        - name: count
          type: int
          default: 0
    graph:
      entry_point: transform
      nodes:
        - name: transform
          type: data_transform
          args:
            expression: '{"count": count + 1}'
      edges: []
"""


@pytest.mark.asyncio
async def test_declarative_agent_set_state():
    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=SET_STATE_YAML,
    )
    graph = await agent.get_graph()
    result = await graph.ainvoke({"messages": [], "result": ""})
    assert result["result"] == "done"


@pytest.mark.asyncio
async def test_declarative_agent_data_transform():
    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=DATA_TRANSFORM_YAML,
    )
    graph = await agent.get_graph()
    result = await graph.ainvoke({"count": 0})
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_declarative_agent_caches_graph():
    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=SET_STATE_YAML,
    )
    graph1 = await agent.get_graph()
    graph2 = await agent.get_graph()
    assert graph1 is graph2


@pytest.mark.asyncio
async def test_declarative_agent_call_llm():
    yaml_content = """\
prompts:
  - id: sys
    version: "1.0.0"
    instructions: "Be helpful"

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
            prompt: 'prompts["sys"]["instructions"]'
      edges: []
"""
    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hi there!"))

    # Inject mock via internal _registries
    registries = RegistryBundle(chat_models={"test-llm": mock_model})

    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=yaml_content,
    )
    agent._registries = registries

    graph = await agent.get_graph()
    result = await graph.ainvoke({"messages": []})

    assert len(result["messages"]) > 0
    mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_declarative_agent_conditional_edges():
    yaml_content = """\
agents:
  test-agent:
    state:
      fields:
        - name: x
          type: int
          default: 0
        - name: result
          type: str
          default: ""
    graph:
      entry_point: check
      nodes:
        - name: check
          type: set_state
          args:
            values:
              result: '"checked"'
        - name: high
          type: set_state
          args:
            values:
              result: '"high"'
        - name: low
          type: set_state
          args:
            values:
              result: '"low"'
      edges:
        - source: check
          branches:
            - condition: "x > 10"
              target: high
          default: low
"""
    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=yaml_content,
    )
    graph = await agent.get_graph()

    result_high = await graph.ainvoke({"x": 15, "result": ""})
    assert result_high["result"] == "high"

    result_low = await graph.ainvoke({"x": 3, "result": ""})
    assert result_low["result"] == "low"


@pytest.mark.asyncio
async def test_declarative_agent_single_agent_auto_match():
    """When config has one agent, it's auto-selected regardless of id."""
    yaml_content = """\
agents:
  only-agent:
    state:
      fields:
        - name: result
          type: str
          default: ""
    graph:
      entry_point: start
      nodes:
        - name: start
          type: set_state
          args:
            values:
              result: '"found"'
      edges: []
"""
    agent = DeclarativeAgent(
        id="different-id",
        version="1.0.0",
        yaml_content=yaml_content,
    )
    graph = await agent.get_graph()
    result = await graph.ainvoke({"result": ""})
    assert result["result"] == "found"
