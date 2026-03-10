"""Tests for declarative schema Pydantic models."""

from __future__ import annotations

from sherma.langgraph.declarative.schema import (
    AgentDef,
    BranchDef,
    CallLLMArgs,
    DataTransformArgs,
    DeclarativeConfig,
    EdgeDef,
    GraphDef,
    InterruptArgs,
    LLMDef,
    NodeDef,
    PromptDef,
    RegistryRef,
    SetStateArgs,
    StateDef,
    StateFieldDef,
    ToolDef,
    ToolNodeArgs,
)


def test_registry_ref_defaults():
    ref = RegistryRef(id="my-tool")
    assert ref.id == "my-tool"
    assert ref.version == "*"


def test_state_field_def():
    field = StateFieldDef(name="messages", type="list", default=[])
    assert field.name == "messages"
    assert field.type == "list"
    assert field.default == []


def test_call_llm_args():
    args = CallLLMArgs(
        llm=RegistryRef(id="gpt-4"),
        prompt='prompts["sys"].instructions',
        tools=[RegistryRef(id="get_weather", version="1.0.0")],
    )
    assert args.llm.id == "gpt-4"
    assert args.tools is not None
    assert len(args.tools) == 1


def test_tool_node_args():
    args = ToolNodeArgs(tools=[RegistryRef(id="tool1")])
    assert len(args.tools) == 1


def test_data_transform_args():
    args = DataTransformArgs(expression='{"result": "done"}')
    assert args.expression == '{"result": "done"}'


def test_set_state_args():
    args = SetStateArgs(values={"count": "count + 1"})
    assert args.values["count"] == "count + 1"


def test_node_def():
    node = NodeDef(
        name="agent",
        type="call_llm",
        args=CallLLMArgs(
            llm=RegistryRef(id="gpt-4"),
            prompt='"hello"',
        ),
    )
    assert node.name == "agent"
    assert node.type == "call_llm"


def test_edge_def_static():
    edge = EdgeDef(source="a", target="b")
    assert edge.source == "a"
    assert edge.target == "b"
    assert edge.branches is None


def test_edge_def_conditional():
    edge = EdgeDef(
        source="agent",
        branches=[BranchDef(condition="true", target="tools")],
        default="__end__",
    )
    assert len(edge.branches) == 1
    assert edge.default == "__end__"


def test_graph_def():
    graph = GraphDef(
        entry_point="agent",
        nodes=[
            NodeDef(
                name="agent",
                type="set_state",
                args=SetStateArgs(values={"x": '"hello"'}),
            )
        ],
        edges=[EdgeDef(source="agent", target="__end__")],
    )
    assert graph.entry_point == "agent"
    assert len(graph.nodes) == 1


def test_declarative_config():
    config = DeclarativeConfig(
        llms=[LLMDef(id="gpt-4", version="1.0.0", model_name="gpt-4")],
        prompts=[PromptDef(id="sys", version="1.0.0", instructions="hello")],
        tools=[ToolDef(id="tool1", version="1.0.0")],
    )
    assert len(config.llms) == 1
    assert len(config.prompts) == 1
    assert len(config.tools) == 1
    assert config.agents == {}


def test_declarative_config_with_agent():
    config = DeclarativeConfig(
        agents={
            "my-agent": AgentDef(
                state=StateDef(
                    fields=[StateFieldDef(name="messages", type="list", default=[])]
                ),
                graph=GraphDef(
                    entry_point="start",
                    nodes=[
                        NodeDef(
                            name="start",
                            type="set_state",
                            args=SetStateArgs(values={"x": '"hi"'}),
                        )
                    ],
                    edges=[],
                ),
            )
        }
    )
    assert "my-agent" in config.agents
    assert config.agents["my-agent"].graph.entry_point == "start"


def test_interrupt_args():
    args = InterruptArgs(value='"What is your name?"')
    assert args.value == '"What is your name?"'


def test_node_def_interrupt():
    node = NodeDef(
        name="ask",
        type="interrupt",
        args=InterruptArgs(value='"What is your name?"'),
    )
    assert node.name == "ask"
    assert node.type == "interrupt"
    assert isinstance(node.args, InterruptArgs)
