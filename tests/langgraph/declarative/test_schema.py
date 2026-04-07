"""Tests for declarative schema Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sherma.langgraph.declarative.schema import (
    AgentDef,
    BranchDef,
    CallLLMArgs,
    DataTransformArgs,
    DeclarativeConfig,
    EdgeDef,
    GraphDef,
    HookDef,
    InterruptArgs,
    LLMDef,
    LoadSkillsArgs,
    NodeDef,
    PromptDef,
    PromptMessageDef,
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
        prompt=[
            PromptMessageDef(role="system", content='prompts["sys"].instructions'),
        ],
        tools=[RegistryRef(id="get_weather", version="1.0.0")],
        state_updates={"messages": "[llm_response]"},
    )
    assert args.llm.id == "gpt-4"
    assert args.tools is not None
    assert len(args.tools) == 1
    assert len(args.prompt) == 1
    assert args.prompt[0].role == "system"


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
            prompt=[PromptMessageDef(role="system", content='"hello"')],
            state_updates={"messages": "[llm_response]"},
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


def test_graph_def_start_edge_without_entry_point():
    graph = GraphDef(
        nodes=[
            NodeDef(
                name="agent",
                type="set_state",
                args=SetStateArgs(values={"x": '"hello"'}),
            )
        ],
        edges=[EdgeDef(source="__start__", target="agent")],
    )
    assert graph.entry_point is None
    assert graph.edges[0].source == "__start__"


def test_graph_def_conditional_start_edge():
    graph = GraphDef(
        nodes=[
            NodeDef(
                name="a",
                type="set_state",
                args=SetStateArgs(values={"x": '"a"'}),
            ),
            NodeDef(
                name="b",
                type="set_state",
                args=SetStateArgs(values={"x": '"b"'}),
            ),
        ],
        edges=[
            EdgeDef(
                source="__start__",
                branches=[BranchDef(condition="true", target="a")],
                default="b",
            )
        ],
    )
    assert graph.entry_point is None
    assert graph.edges[0].branches is not None


def test_graph_def_requires_entry_or_start_edge():
    with pytest.raises(ValidationError, match="entry_point"):
        GraphDef(
            nodes=[
                NodeDef(
                    name="a",
                    type="set_state",
                    args=SetStateArgs(values={"x": '"a"'}),
                )
            ],
            edges=[EdgeDef(source="a", target="__end__")],
        )


def test_graph_def_rejects_both_entry_and_start_edge():
    with pytest.raises(ValidationError, match="cannot have both"):
        GraphDef(
            entry_point="a",
            nodes=[
                NodeDef(
                    name="a",
                    type="set_state",
                    args=SetStateArgs(values={"x": '"a"'}),
                )
            ],
            edges=[EdgeDef(source="__start__", target="a")],
        )


def test_declarative_config():
    config = DeclarativeConfig(
        manifest_version=1,
        llms=[LLMDef(id="gpt-4", version="1.0.0", model_name="gpt-4")],
        prompts=[PromptDef(id="sys", version="1.0.0", instructions="hello")],
        tools=[ToolDef(id="tool1", version="1.0.0")],
    )
    assert len(config.llms) == 1
    assert len(config.prompts) == 1
    assert len(config.tools) == 1
    assert config.agents == {}
    assert config.manifest_version == 1


def test_declarative_config_with_agent():
    config = DeclarativeConfig(
        manifest_version=1,
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
        },
    )
    assert "my-agent" in config.agents
    assert config.agents["my-agent"].graph.entry_point == "start"


def test_interrupt_args_requires_value():
    with pytest.raises(ValidationError):
        InterruptArgs()


def test_interrupt_args_with_value():
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


def test_response_format_def():
    from sherma.langgraph.declarative.schema import ResponseFormatDef

    rf = ResponseFormatDef(
        name="UserInfo",
        description="Extracted user info",
        **{"schema": {"type": "object", "properties": {"name": {"type": "string"}}}},
    )
    assert rf.name == "UserInfo"
    assert rf.description == "Extracted user info"
    assert rf.schema_["type"] == "object"


def test_prompt_message_def():
    msg = PromptMessageDef(role="system", content='"You are helpful"')
    assert msg.role == "system"
    assert msg.content == '"You are helpful"'


def test_prompt_message_def_messages_role():
    msg = PromptMessageDef(role="messages", content="state.messages")
    assert msg.role == "messages"


def test_prompt_message_def_invalid_role():
    with pytest.raises(ValidationError):
        PromptMessageDef(role="invalid", content='"test"')


def test_call_llm_args_array_prompt():
    args = CallLLMArgs(
        llm=RegistryRef(id="gpt-4"),
        prompt=[
            PromptMessageDef(role="system", content='"You are helpful"'),
            PromptMessageDef(role="messages", content="state.messages"),
            PromptMessageDef(role="human", content='"Summarize"'),
        ],
        state_updates={"messages": "[llm_response]"},
    )
    assert len(args.prompt) == 3
    assert args.prompt[0].role == "system"
    assert args.prompt[1].role == "messages"
    assert args.prompt[2].role == "human"


def test_call_llm_args_with_response_format():
    from sherma.langgraph.declarative.schema import ResponseFormatDef

    args = CallLLMArgs(
        llm=RegistryRef(id="gpt-4"),
        prompt=[PromptMessageDef(role="system", content='"Extract info"')],
        state_updates={"messages": "[llm_response]"},
        response_format=ResponseFormatDef(
            name="UserInfo",
            **{
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
        ),
    )
    assert args.response_format is not None
    assert args.response_format.name == "UserInfo"
    assert args.response_format.schema_["required"] == ["name"]


def test_call_llm_args_response_format_default_none():
    args = CallLLMArgs(
        llm=RegistryRef(id="gpt-4"),
        prompt=[PromptMessageDef(role="system", content='"hello"')],
        state_updates={"messages": "[llm_response]"},
    )
    assert args.response_format is None


def test_call_llm_args_without_llm():
    """CallLLMArgs.llm defaults to None when omitted."""
    args = CallLLMArgs(
        prompt=[PromptMessageDef(role="system", content='"hello"')],
        state_updates={"messages": "[llm_response]"},
    )
    assert args.llm is None


def test_declarative_config_default_llm():
    """DeclarativeConfig accepts a top-level default_llm."""
    config = DeclarativeConfig(
        manifest_version=1,
        default_llm=RegistryRef(id="gpt-4"),
        llms=[LLMDef(id="gpt-4", model_name="gpt-4")],
    )
    assert config.default_llm is not None
    assert config.default_llm.id == "gpt-4"


def test_declarative_config_default_llm_none_by_default():
    """default_llm is None when not specified."""
    config = DeclarativeConfig(manifest_version=1)
    assert config.default_llm is None


def test_declarative_config_manifest_version_required():
    """manifest_version is required and cannot be omitted."""
    with pytest.raises(ValidationError):
        DeclarativeConfig()


def test_declarative_config_manifest_version_must_be_int():
    """manifest_version must be an integer."""
    with pytest.raises(ValidationError):
        DeclarativeConfig(manifest_version="one")


def test_hook_def_with_import_path():
    hook = HookDef(import_path="my_module.MyHook")
    assert hook.import_path == "my_module.MyHook"
    assert hook.url is None


def test_hook_def_with_url():
    hook = HookDef(url="http://localhost:8080/hooks")
    assert hook.url == "http://localhost:8080/hooks"
    assert hook.import_path is None


def test_hook_def_requires_one_source():
    with pytest.raises(ValidationError, match="requires either"):
        HookDef()


def test_hook_def_rejects_both_sources():
    with pytest.raises(ValidationError, match="cannot have both"):
        HookDef(import_path="my_module.MyHook", url="http://localhost:8080/hooks")


# --- use_sub_agents_as_tools type normalization ---


def test_use_sub_agents_as_tools_true_becomes_all():
    """YAML `true` is normalized to `"all"`."""
    args = CallLLMArgs(
        prompt=[PromptMessageDef(role="system", content='"hello"')],
        state_updates={"messages": "[llm_response]"},
        use_sub_agents_as_tools=True,
    )
    assert args.use_sub_agents_as_tools == "all"


def test_use_sub_agents_as_tools_false_stays_false():
    args = CallLLMArgs(
        prompt=[PromptMessageDef(role="system", content='"hello"')],
        state_updates={"messages": "[llm_response]"},
        use_sub_agents_as_tools=False,
    )
    assert args.use_sub_agents_as_tools is False


def test_use_sub_agents_as_tools_all_string():
    args = CallLLMArgs(
        prompt=[PromptMessageDef(role="system", content='"hello"')],
        state_updates={"messages": "[llm_response]"},
        use_sub_agents_as_tools="all",
    )
    assert args.use_sub_agents_as_tools == "all"


def test_use_sub_agents_as_tools_list_of_refs():
    args = CallLLMArgs(
        prompt=[PromptMessageDef(role="system", content='"hello"')],
        state_updates={"messages": "[llm_response]"},
        use_sub_agents_as_tools=[
            {"id": "weather-agent", "version": "1.0.0"},
            {"id": "search-agent", "version": "1.0.0"},
        ],
    )
    assert isinstance(args.use_sub_agents_as_tools, list)
    assert len(args.use_sub_agents_as_tools) == 2
    assert args.use_sub_agents_as_tools[0].id == "weather-agent"
    assert args.use_sub_agents_as_tools[1].id == "search-agent"


def test_use_sub_agents_as_tools_invalid_value():
    with pytest.raises(ValidationError):
        CallLLMArgs(
            prompt=[PromptMessageDef(role="system", content='"hello"')],
            state_updates={"messages": "[llm_response]"},
            use_sub_agents_as_tools="invalid",
        )


def test_use_sub_agents_as_tools_default_is_false():
    args = CallLLMArgs(
        prompt=[PromptMessageDef(role="system", content='"hello"')],
        state_updates={"messages": "[llm_response]"},
    )
    assert args.use_sub_agents_as_tools is False


def test_load_skills_args():
    args = LoadSkillsArgs(skill_ids='[{"id": "weather", "version": "1.0.0"}]')
    assert args.skill_ids == '[{"id": "weather", "version": "1.0.0"}]'


def test_load_skills_node_def_parsing():
    """NodeDef with type=load_skills resolves args to LoadSkillsArgs."""
    node = NodeDef(
        name="load",
        type="load_skills",
        args={"skill_ids": "state.selected_skills"},
    )
    assert isinstance(node.args, LoadSkillsArgs)
    assert node.args.skill_ids == "state.selected_skills"
