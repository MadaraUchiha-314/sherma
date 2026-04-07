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
manifest_version: 1

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
manifest_version: 1

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
            expression: '{"count": state.count + 1}'
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
    config = {"configurable": {"thread_id": "t1"}}
    result = await graph.ainvoke({"messages": [], "result": ""}, config)
    assert result["result"] == "done"


@pytest.mark.asyncio
async def test_declarative_agent_data_transform():
    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=DATA_TRANSFORM_YAML,
    )
    graph = await agent.get_graph()
    config = {"configurable": {"thread_id": "t1"}}
    result = await graph.ainvoke({"count": 0}, config)
    assert result["count"] == 1


START_EDGE_STATIC_YAML = """\
manifest_version: 1

agents:
  test-agent:
    state:
      fields:
        - name: result
          type: str
          default: ""
    graph:
      nodes:
        - name: setter
          type: set_state
          args:
            values:
              result: '"from-start-edge"'
      edges:
        - source: __start__
          target: setter
        - source: setter
          target: __end__
"""


START_EDGE_CONDITIONAL_YAML = """\
manifest_version: 1

agents:
  test-agent:
    state:
      fields:
        - name: route
          type: str
          default: ""
        - name: result
          type: str
          default: ""
    graph:
      nodes:
        - name: a
          type: set_state
          args:
            values:
              result: '"took-a"'
        - name: b
          type: set_state
          args:
            values:
              result: '"took-b"'
      edges:
        - source: __start__
          branches:
            - condition: 'state.route == "a"'
              target: a
          default: b
        - source: a
          target: __end__
        - source: b
          target: __end__
"""


@pytest.mark.asyncio
async def test_declarative_agent_static_start_edge():
    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=START_EDGE_STATIC_YAML,
    )
    graph = await agent.get_graph()
    config = {"configurable": {"thread_id": "t1"}}
    result = await graph.ainvoke({"result": ""}, config)
    assert result["result"] == "from-start-edge"


@pytest.mark.asyncio
async def test_declarative_agent_conditional_start_edge_a():
    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=START_EDGE_CONDITIONAL_YAML,
    )
    graph = await agent.get_graph()
    config = {"configurable": {"thread_id": "t1"}}
    result = await graph.ainvoke({"route": "a", "result": ""}, config)
    assert result["result"] == "took-a"


@pytest.mark.asyncio
async def test_declarative_agent_conditional_start_edge_default():
    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=START_EDGE_CONDITIONAL_YAML,
    )
    graph = await agent.get_graph()
    config = {"configurable": {"thread_id": "t2"}}
    result = await graph.ainvoke({"route": "other", "result": ""}, config)
    assert result["result"] == "took-b"


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
manifest_version: 1

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
            prompt:
              - role: system
                content: 'prompts["sys"]["instructions"]'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
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
    config = {"configurable": {"thread_id": "t1"}}
    result = await graph.ainvoke({"messages": []}, config)

    assert len(result["messages"]) > 0
    mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_declarative_agent_conditional_edges():
    yaml_content = """\
manifest_version: 1

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
            - condition: "state.x > 10"
              target: high
          default: low
"""
    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=yaml_content,
    )
    graph = await agent.get_graph()

    config_high = {"configurable": {"thread_id": "t-high"}}
    result_high = await graph.ainvoke({"x": 15, "result": ""}, config_high)
    assert result_high["result"] == "high"

    config_low = {"configurable": {"thread_id": "t-low"}}
    result_low = await graph.ainvoke({"x": 3, "result": ""}, config_low)
    assert result_low["result"] == "low"


@pytest.mark.asyncio
async def test_declarative_agent_single_agent_auto_match():
    """When config has one agent, it's auto-selected regardless of id."""
    yaml_content = """\
manifest_version: 1

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
    config = {"configurable": {"thread_id": "t1"}}
    result = await graph.ainvoke({"result": ""}, config)
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
manifest_version: 1

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


_APPROVAL_CONDITION = (
    "state.messages[size(state.messages) - 1]"
    '["additional_kwargs"]["decision"] == "approve"'
)

APPROVAL_ROUTING_YAML = f"""\
manifest_version: 1

agents:
  test-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
        - name: outcome
          type: str
          default: ""
    graph:
      entry_point: ask_approval
      nodes:
        - name: ask_approval
          type: interrupt
          args:
            value: '"Approve or reject?"'
        - name: handle_approved
          type: set_state
          args:
            values:
              outcome: '"approved"'
        - name: handle_rejected
          type: set_state
          args:
            values:
              outcome: '"rejected"'
      edges:
        - source: ask_approval
          branches:
            - condition: '{_APPROVAL_CONDITION}'
              target: handle_approved
          default: handle_rejected
        - source: handle_approved
          target: __end__
        - source: handle_rejected
          target: __end__
"""


@pytest.mark.asyncio
async def test_declarative_agent_additional_kwargs_routing_approve():
    """CEL routes on additional_kwargs set by a hook on interrupt response."""
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    from sherma.hooks.executor import BaseHookExecutor
    from sherma.hooks.types import NodeExitContext

    class TagDecisionHook(BaseHookExecutor):
        async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
            if ctx.node_type != "interrupt":
                return None
            msgs = ctx.result.get("messages", [])
            if msgs and isinstance(msgs[-1], HumanMessage):
                content = str(msgs[-1].content).strip().lower()
                decision = "approve" if "approve" in content else "reject"
                ctx.result["messages"] = [
                    HumanMessage(
                        content=msgs[-1].content,
                        additional_kwargs={"decision": decision},
                    )
                ]
            return ctx

    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=APPROVAL_ROUTING_YAML,
        hooks=[TagDecisionHook()],
    )
    compiled = await agent.get_graph()
    config = {"configurable": {"thread_id": "approve-test"}}

    # First invoke triggers interrupt
    await compiled.ainvoke({"messages": []}, config)

    # Resume with "approve" — hook tags it, CEL routes to handle_approved
    result = await compiled.ainvoke(Command(resume="approve"), config)
    assert result["outcome"] == "approved"

    # Verify the human message carries additional_kwargs
    human_msgs = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    tagged = [m for m in human_msgs if m.additional_kwargs.get("decision")]
    assert len(tagged) == 1
    assert tagged[0].additional_kwargs["decision"] == "approve"


@pytest.mark.asyncio
async def test_declarative_agent_additional_kwargs_routing_reject():
    """CEL routes to default when additional_kwargs decision is not 'approve'."""
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    from sherma.hooks.executor import BaseHookExecutor
    from sherma.hooks.types import NodeExitContext

    class TagDecisionHook(BaseHookExecutor):
        async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
            if ctx.node_type != "interrupt":
                return None
            msgs = ctx.result.get("messages", [])
            if msgs and isinstance(msgs[-1], HumanMessage):
                content = str(msgs[-1].content).strip().lower()
                decision = "approve" if "approve" in content else "reject"
                ctx.result["messages"] = [
                    HumanMessage(
                        content=msgs[-1].content,
                        additional_kwargs={"decision": decision},
                    )
                ]
            return ctx

    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=APPROVAL_ROUTING_YAML,
        hooks=[TagDecisionHook()],
    )
    compiled = await agent.get_graph()
    config = {"configurable": {"thread_id": "reject-test"}}

    await compiled.ainvoke({"messages": []}, config)

    # Resume with "no thanks" — hook tags as reject, CEL falls to default
    result = await compiled.ainvoke(Command(resume="no thanks"), config)
    assert result["outcome"] == "rejected"


@pytest.mark.asyncio
async def test_declarative_agent_with_config_object():
    """DeclarativeAgent accepts a pre-built DeclarativeConfig."""
    config = DeclarativeConfig(
        manifest_version=1,
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
    config = {"configurable": {"thread_id": "t1"}}
    result = await graph.ainvoke({"result": ""}, config)
    assert result["result"] == "from_config"


DEFAULT_LLM_YAML = """\
manifest_version: 1

default_llm:
  id: test-llm

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
            prompt:
              - role: system
                content: 'prompts["sys"]["instructions"]'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
      edges: []
"""

DEFAULT_LLM_OVERRIDE_YAML = """\
manifest_version: 1

default_llm:
  id: default-llm

llms:
  - id: default-llm
    version: "1.0.0"
    provider: openai
    model_name: default
  - id: override-llm
    version: "1.0.0"
    provider: openai
    model_name: override

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
              id: override-llm
            prompt:
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
      edges: []
"""

NO_LLM_YAML = """\
manifest_version: 1

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
            prompt:
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
      edges: []
"""


@pytest.mark.asyncio
async def test_declarative_agent_default_llm():
    """call_llm node falls back to default_llm when llm is omitted."""
    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))

    registries = RegistryBundle(chat_models={"test-llm": mock_model})

    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=DEFAULT_LLM_YAML,
    )
    agent._registries = registries

    graph = await agent.get_graph()
    config = {"configurable": {"thread_id": "t1"}}
    result = await graph.ainvoke({"messages": []}, config)

    assert len(result["messages"]) > 0
    mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_declarative_agent_step_llm_overrides_default():
    """Step-level llm takes precedence over default_llm."""
    mock_override = AsyncMock()
    mock_override.ainvoke = AsyncMock(return_value=AIMessage(content="Override!"))
    mock_default = AsyncMock()
    mock_default.ainvoke = AsyncMock(return_value=AIMessage(content="Default!"))

    registries = RegistryBundle(
        chat_models={
            "override-llm": mock_override,
            "default-llm": mock_default,
        }
    )

    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=DEFAULT_LLM_OVERRIDE_YAML,
    )
    agent._registries = registries

    graph = await agent.get_graph()
    config = {"configurable": {"thread_id": "t1"}}
    await graph.ainvoke({"messages": []}, config)

    # Override model should have been called, not the default
    mock_override.ainvoke.assert_called_once()
    mock_default.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_declarative_agent_no_llm_raises():
    """Missing both step-level llm and default_llm raises GraphConstructionError."""
    from sherma.exceptions import GraphConstructionError

    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=NO_LLM_YAML,
    )

    with pytest.raises(GraphConstructionError, match="no 'llm'"):
        await agent.get_graph()


MULTI_NODE_DEFAULT_LLM_YAML = """\
manifest_version: 1

default_llm:
  id: shared-llm

llms:
  - id: shared-llm
    version: "1.0.0"
    provider: openai
    model_name: shared

agents:
  test-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
    graph:
      entry_point: node_a
      nodes:
        - name: node_a
          type: call_llm
          args:
            prompt:
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
        - name: node_b
          type: call_llm
          args:
            prompt:
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
      edges:
        - source: node_a
          target: node_b
"""


@pytest.mark.asyncio
async def test_declarative_agent_default_llm_multiple_nodes():
    """Multiple call_llm nodes all inherit default_llm."""
    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Reply"))

    registries = RegistryBundle(chat_models={"shared-llm": mock_model})

    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=MULTI_NODE_DEFAULT_LLM_YAML,
    )
    agent._registries = registries

    graph = await agent.get_graph()
    config = {"configurable": {"thread_id": "t1"}}
    await graph.ainvoke({"messages": []}, config)

    # Both nodes should have called the same shared model
    assert mock_model.ainvoke.call_count == 2


CHECKPOINTER_YAML = """\
manifest_version: 1

checkpointer:
  type: memory

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
async def test_declarative_agent_yaml_checkpointer():
    """Checkpointer declared in YAML config is wired into the compiled graph."""
    from langgraph.types import Command

    agent = DeclarativeAgent(
        id="test-agent",
        version="1.0.0",
        yaml_content=CHECKPOINTER_YAML,
    )
    compiled = await agent.get_graph()

    # The checkpointer should be set from YAML config
    assert compiled.checkpointer is not None

    config = {"configurable": {"thread_id": "t-yaml"}}
    await compiled.ainvoke({"messages": []}, config)

    state = await compiled.aget_state(config)
    assert len(state.tasks) > 0

    result = await compiled.ainvoke(Command(resume="Bob"), config)
    from langchain_core.messages import HumanMessage

    human_msgs = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    assert any(m.content == "Bob" for m in human_msgs)
