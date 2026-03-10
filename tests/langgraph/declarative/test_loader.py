"""Tests for declarative config loader."""

from __future__ import annotations

import pytest

from sherma.exceptions import DeclarativeConfigError
from sherma.langgraph.declarative.loader import (
    RegistryBundle,
    import_tool,
    load_declarative_config,
    populate_registries,
    validate_config,
)

MINIMAL_YAML = """\
prompts:
  - id: sys
    version: "1.0.0"
    instructions: "Be helpful"

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

agents:
  my-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
    graph:
      entry_point: start
      nodes:
        - name: start
          type: set_state
          args:
            values:
              x: '"hello"'
      edges: []
"""

CALL_LLM_YAML = """\
prompts:
  - id: sys
    version: "1.0.0"
    instructions: "Be helpful"

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

tools:
  - id: get_weather
    version: "1.0.0"

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
              id: gpt-4
              version: "1.0.0"
            prompt: 'prompts["sys"].instructions'
            tools:
              - id: get_weather
                version: "1.0.0"
        - name: tools
          type: tool_node
          args: {}  # auto-inherits tools from call_llm
      edges:
        - source: agent
          branches:
            - condition: 'size(messages) > 0'
              target: tools
          default: __end__
        - source: tools
          target: agent
"""


def test_load_from_content():
    config = load_declarative_config(yaml_content=MINIMAL_YAML)
    assert "my-agent" in config.agents
    assert len(config.prompts) == 1
    assert len(config.llms) == 1


def test_load_both_path_and_content_raises():
    with pytest.raises(DeclarativeConfigError, match="not both"):
        load_declarative_config(yaml_path="x.yaml", yaml_content="y")


def test_load_neither_raises():
    with pytest.raises(DeclarativeConfigError, match="Provide either"):
        load_declarative_config()


def test_load_missing_file():
    with pytest.raises(DeclarativeConfigError, match="not found"):
        load_declarative_config(yaml_path="/nonexistent/file.yaml")


def test_load_invalid_yaml():
    with pytest.raises(DeclarativeConfigError, match="Invalid YAML"):
        load_declarative_config(yaml_content="{{invalid: yaml: [")


def test_load_non_mapping_yaml():
    with pytest.raises(DeclarativeConfigError, match="root must be a mapping"):
        load_declarative_config(yaml_content="- just a list")


def test_load_call_llm_config():
    config = load_declarative_config(yaml_content=CALL_LLM_YAML)
    agent = config.agents["weather-agent"]
    assert agent.graph.entry_point == "agent"
    assert len(agent.graph.nodes) == 2
    assert agent.graph.nodes[0].type == "call_llm"
    assert agent.graph.nodes[1].type == "tool_node"


@pytest.mark.asyncio
async def test_populate_registries():
    config = load_declarative_config(yaml_content=MINIMAL_YAML)
    # Pre-populate chat_models to skip auto-creation (avoids needing API key)
    from unittest.mock import MagicMock

    registries = RegistryBundle(chat_models={"gpt-4": MagicMock()})
    await populate_registries(config, registries)

    llm = await registries.llm_registry.get("gpt-4")
    assert llm.model_name == "gpt-4"

    prompt = await registries.prompt_registry.get("sys")
    assert prompt.instructions == "Be helpful"

    # Pre-populated chat model should be preserved
    assert "gpt-4" in registries.chat_models


def test_validate_config_entry_point_missing():
    config = load_declarative_config(yaml_content=MINIMAL_YAML)
    config.agents["my-agent"].graph.entry_point = "nonexistent"
    with pytest.raises(DeclarativeConfigError, match="Entry point"):
        validate_config(config, "my-agent")


def test_validate_config_edge_source_missing():
    config = load_declarative_config(yaml_content=MINIMAL_YAML)
    from sherma.langgraph.declarative.schema import EdgeDef

    config.agents["my-agent"].graph.edges.append(
        EdgeDef(source="nonexistent", target="start")
    )
    with pytest.raises(DeclarativeConfigError, match="Edge source"):
        validate_config(config, "my-agent")


def test_validate_config_agent_not_found():
    config = load_declarative_config(yaml_content=MINIMAL_YAML)
    with pytest.raises(DeclarativeConfigError, match="not found in config"):
        validate_config(config, "nonexistent-agent")


def test_validate_call_llm_without_messages():
    yaml_content = """\
llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

agents:
  bad-agent:
    state:
      fields:
        - name: data
          type: str
    graph:
      entry_point: agent
      nodes:
        - name: agent
          type: call_llm
          args:
            llm:
              id: gpt-4
            prompt: '"hello"'
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    with pytest.raises(DeclarativeConfigError, match="messages"):
        validate_config(config, "bad-agent")


def test_validate_call_llm_tools_without_tool_node():
    yaml_content = """\
llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

agents:
  bad-agent:
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
              id: gpt-4
            prompt: '"hello"'
            tools:
              - id: some-tool
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    with pytest.raises(DeclarativeConfigError, match="tool_node"):
        validate_config(config, "bad-agent")


def test_import_tool_invalid_path():
    with pytest.raises(DeclarativeConfigError, match="Invalid import_path"):
        import_tool("no_dots")


def test_import_tool_missing_module():
    with pytest.raises(DeclarativeConfigError, match="Cannot import"):
        import_tool("nonexistent.module.tool")


def test_import_tool_missing_attr():
    with pytest.raises(DeclarativeConfigError, match="has no attribute"):
        import_tool("os.path.nonexistent_tool")


def test_import_tool_not_a_tool():
    with pytest.raises(DeclarativeConfigError, match="not a LangGraph BaseTool"):
        import_tool("os.path.join")


def test_llm_provider_default():
    config = load_declarative_config(yaml_content=MINIMAL_YAML)
    assert config.llms[0].provider == "openai"


def test_tool_import_path_parsed():
    yaml_content = """\
tools:
  - id: my-tool
    version: "1.0.0"
    import_path: some.module.my_tool

agents:
  a:
    state:
      fields: []
    graph:
      entry_point: start
      nodes:
        - name: start
          type: set_state
          args:
            values:
              x: '"hi"'
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    assert config.tools[0].import_path == "some.module.my_tool"


@pytest.mark.asyncio
async def test_populate_auto_imports_tools():
    """Tools with import_path are auto-imported during populate."""
    yaml_content = """\
tools:
  - id: get_weather
    version: "1.0.0"
    import_path: examples.tools.get_weather

agents:
  a:
    state:
      fields: []
    graph:
      entry_point: start
      nodes:
        - name: start
          type: set_state
          args:
            values:
              x: '"hi"'
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    registries = RegistryBundle()
    await populate_registries(config, registries)

    tool = await registries.tool_registry.get("get_weather")
    assert tool.id == "get_weather"


@pytest.mark.asyncio
async def test_populate_skill_cards_local(tmp_path):
    """Skill cards with skill_card_path are loaded during populate."""
    import json

    card_data = {
        "name": "Test Skill",
        "description": "A test skill",
        "base_uri": str(tmp_path),
        "files": ["SKILL.md"],
    }
    card_file = tmp_path / "skill-card.json"
    card_file.write_text(json.dumps(card_data))

    yaml_content = f"""\
skills:
  - id: test-skill
    version: "1.0.0"
    skill_card_path: "{card_file}"

agents:
  a:
    state:
      fields: []
    graph:
      entry_point: start
      nodes:
        - name: start
          type: set_state
          args:
            values:
              x: '"hi"'
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    registries = RegistryBundle()
    await populate_registries(config, registries)

    skill = await registries.skill_registry.get("test-skill", "==1.0.0")
    assert skill.skill_card is not None
    assert skill.skill_card.name == "Test Skill"
    assert skill.skill_card.files == ["SKILL.md"]


@pytest.mark.asyncio
async def test_populate_skill_cards_remote():
    """Skill cards with url are fetched and registered as Skill with skill_card."""
    from unittest.mock import AsyncMock, MagicMock, patch

    yaml_content = """\
skills:
  - id: remote-skill
    version: "1.0.0"
    url: "https://example.com/skill-card.json"

agents:
  a:
    state:
      fields: []
    graph:
      entry_point: start
      nodes:
        - name: start
          type: set_state
          args:
            values:
              x: '"hi"'
      edges: []
"""
    remote_data = {
        "name": "Remote Skill",
        "description": "Fetched remotely",
        "base_uri": "https://example.com/skill",
        "files": ["SKILL.md"],
    }
    mock_response = MagicMock()
    mock_response.json.return_value = remote_data
    mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    async def fake_get_http_client(*a, **kw):
        return mock_client

    config = load_declarative_config(yaml_content=yaml_content)
    registries = RegistryBundle()

    with patch(
        "sherma.registry.skill_card.get_http_client",
        side_effect=fake_get_http_client,
    ):
        await populate_registries(config, registries)

    # The skill should be registered with a skill_card
    assert "remote-skill" in registries.skill_registry._entries
    skill = await registries.skill_registry.get("remote-skill", "==1.0.0")
    assert skill.skill_card is not None
    assert skill.skill_card.name == "Remote Skill"


def test_load_use_tools_from_registry_config():
    """YAML with use_tools_from_registry: true parses correctly."""
    yaml_content = """\
llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

agents:
  agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
    graph:
      entry_point: llm
      nodes:
        - name: llm
          type: call_llm
          args:
            llm:
              id: gpt-4
            prompt: '"hello"'
            use_tools_from_registry: true
        - name: tools
          type: tool_node
      edges:
        - source: llm
          branches:
            - condition: has_tool_calls
              target: tools
          default: __end__
        - source: tools
          target: llm
"""
    from sherma.langgraph.declarative.schema import CallLLMArgs, ToolNodeArgs

    config = load_declarative_config(yaml_content=yaml_content)
    agent = config.agents["agent"]
    llm_node = agent.graph.nodes[0]
    tool_node = agent.graph.nodes[1]

    assert isinstance(llm_node.args, CallLLMArgs)
    assert llm_node.args.use_tools_from_registry is True
    assert llm_node.args.tools is None

    assert isinstance(tool_node.args, ToolNodeArgs)
    assert tool_node.args.tools is None


def test_load_use_tools_from_loaded_skills_config():
    """YAML with use_tools_from_loaded_skills: true parses correctly."""
    yaml_content = """\
llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

agents:
  agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
    graph:
      entry_point: llm
      nodes:
        - name: llm
          type: call_llm
          args:
            llm:
              id: gpt-4
            prompt: '"hello"'
            use_tools_from_loaded_skills: true
        - name: tools
          type: tool_node
      edges: []
"""
    from sherma.langgraph.declarative.schema import CallLLMArgs

    config = load_declarative_config(yaml_content=yaml_content)
    agent = config.agents["agent"]
    llm_node = agent.graph.nodes[0]

    assert isinstance(llm_node.args, CallLLMArgs)
    assert llm_node.args.use_tools_from_loaded_skills is True


def test_validate_registry_tools_with_explicit_tools():
    """use_tools_from_registry + explicit tools on call_llm is an error."""
    yaml_content = """\
llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

agents:
  bad-agent:
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
              id: gpt-4
            prompt: '"hello"'
            use_tools_from_registry: true
            tools:
              - id: some-tool
        - name: tools
          type: tool_node
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    with pytest.raises(DeclarativeConfigError, match="cannot specify both"):
        validate_config(config, "bad-agent")


def test_validate_both_registry_and_loaded_skills():
    """use_tools_from_registry + use_tools_from_loaded_skills is an error."""
    yaml_content = """\
llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

agents:
  bad-agent:
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
              id: gpt-4
            prompt: '"hello"'
            use_tools_from_registry: true
            use_tools_from_loaded_skills: true
        - name: tools
          type: tool_node
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    with pytest.raises(DeclarativeConfigError, match="cannot specify both"):
        validate_config(config, "bad-agent")


def test_validate_tools_without_tool_node():
    """call_llm with tool options requires a tool_node in the graph."""
    yaml_content = """\
llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

agents:
  bad-agent:
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
              id: gpt-4
            prompt: '"hello"'
            use_tools_from_registry: true
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    with pytest.raises(DeclarativeConfigError, match="tool_node"):
        validate_config(config, "bad-agent")


def test_skill_def_with_skill_card_path():
    yaml_content = """\
skills:
  - id: my-skill
    version: "1.0.0"
    skill_card_path: "/path/to/card.json"

agents:
  a:
    state:
      fields: []
    graph:
      entry_point: start
      nodes:
        - name: start
          type: set_state
          args:
            values:
              x: '"hi"'
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    assert config.skills[0].skill_card_path == "/path/to/card.json"
