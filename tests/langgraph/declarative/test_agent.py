"""Tests for DeclarativeAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from sherma.langgraph.declarative.agent import (
    DeclarativeAgent,
    _build_state_class,
)
from sherma.langgraph.declarative.loader import RegistryBundle
from sherma.langgraph.declarative.schema import (
    AgentDef,
    DeclarativeConfig,
    GraphDef,
    NodeDef,
    SetStateArgs,
    StateDef,
    StateFieldDef,
)

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


def test_state_class_injects_internal_state():
    """__sherma__ is auto-injected when has_skills=True."""
    from sherma.langgraph.declarative.nodes import INTERNAL_STATE_KEY

    agent_def = AgentDef(
        state=StateDef(
            fields=[StateFieldDef(name="messages", type="list")],
        ),
        graph=GraphDef(entry_point="start", nodes=[], edges=[]),
    )

    state_cls = _build_state_class(agent_def, has_skills=True)

    # Should have __sherma__ in annotations
    assert INTERNAL_STATE_KEY in state_cls.__annotations__
    assert state_cls.__annotations__[INTERNAL_STATE_KEY] is dict


def test_state_class_no_injection_without_skills():
    """__sherma__ is NOT injected when has_skills=False."""
    from langgraph.graph import MessagesState

    agent_def = AgentDef(
        state=StateDef(
            fields=[StateFieldDef(name="messages", type="list")],
        ),
        graph=GraphDef(entry_point="start", nodes=[], edges=[]),
    )

    state_cls = _build_state_class(agent_def, has_skills=False)

    # Should be plain MessagesState (no subclass needed)
    assert state_cls is MessagesState


INTERRUPT_YAML = """\
agents:
  test-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
    graph:
      entry_point: ask
      nodes:
        - name: ask
          type: interrupt
          args:
            value: '"What is your name?"'
      edges: []
"""


@pytest.mark.asyncio
async def test_declarative_agent_interrupt_node():
    """DeclarativeAgent with an interrupt node compiles and pauses on invoke."""
    from langgraph.types import Command

    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=INTERRUPT_YAML,
    )
    compiled = await agent.get_graph()

    # Attach a checkpointer so interrupt/resume works
    from langgraph.checkpoint.memory import MemorySaver

    compiled.checkpointer = MemorySaver()

    config = {"configurable": {"thread_id": "t1"}}
    result = await compiled.ainvoke({"messages": []}, config)

    # After interrupt, the graph should have paused — check via get_state
    state = await compiled.aget_state(config)
    # The interrupt value should be surfaced in state.tasks
    assert len(state.tasks) > 0

    # Resume with a value
    result = await compiled.ainvoke(
        Command(resume="Alice"),
        config,
    )
    # The resumed value should appear as a HumanMessage
    from langchain_core.messages import HumanMessage

    human_msgs = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    assert any(m.content == "Alice" for m in human_msgs)


@pytest.mark.asyncio
async def test_declarative_agent_with_config_object():
    """DeclarativeAgent accepts a pre-built DeclarativeConfig."""
    config = DeclarativeConfig(
        agents={
            "test-agent": AgentDef(
                state=StateDef(
                    fields=[
                        StateFieldDef(name="result", type="str", default=""),
                    ],
                ),
                graph=GraphDef(
                    entry_point="setter",
                    nodes=[
                        NodeDef(
                            name="setter",
                            type="set_state",
                            args=SetStateArgs(values={"result": '"from_config"'}),
                        ),
                    ],
                    edges=[],
                ),
            ),
        },
    )
    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        config=config,
    )
    graph = await agent.get_graph()
    result = await graph.ainvoke({"result": ""})
    assert result["result"] == "from_config"
