"""Tests for declarative config loader."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from sherma.exceptions import DeclarativeConfigError
from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.manager import HookManager
from sherma.hooks.remote import RemoteHookExecutor
from sherma.hooks.types import CheckpointerCreateContext
from sherma.langgraph.declarative.loader import (
    RegistryBundle,
    build_checkpointer,
    import_tool,
    load_declarative_config,
    populate_hooks,
    populate_registries,
    validate_config,
)
from sherma.langgraph.declarative.schema import (
    MemoryCheckpointerDef,
    PostgresCheckpointerDef,
    RedisCheckpointerDef,
)

MINIMAL_YAML = """\
manifest_version: 1

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
manifest_version: 1

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
            prompt:
              - role: system
                content: 'prompts["sys"].instructions'
              - role: messages
                content: 'state.messages'
            tools:
              - id: get_weather
                version: "1.0.0"
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
          args: {}  # auto-inherits tools from call_llm
      edges:
        - source: agent
          branches:
            - condition: 'size(state.messages) > 0'
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


def test_load_missing_manifest_version():
    """YAML without manifest_version raises a validation error."""
    yaml_content = """\
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
    with pytest.raises(DeclarativeConfigError, match="manifest_version"):
        load_declarative_config(yaml_content=yaml_content)


def test_load_manifest_version_parsed():
    """manifest_version is correctly parsed from YAML."""
    config = load_declarative_config(yaml_content=MINIMAL_YAML)
    assert config.manifest_version == 1


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


def test_load_default_llm_from_yaml():
    """default_llm parses correctly from YAML."""
    yaml_content = """\
manifest_version: 1

default_llm:
  id: gpt-4

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
      entry_point: agent
      nodes:
        - name: agent
          type: call_llm
          args:
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    assert config.default_llm is not None
    assert config.default_llm.id == "gpt-4"


def test_load_default_llm_absent():
    """default_llm is None when not specified in YAML."""
    config = load_declarative_config(yaml_content=MINIMAL_YAML)
    assert config.default_llm is None


def test_validate_call_llm_without_messages():
    yaml_content = """\
manifest_version: 1

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    with pytest.raises(DeclarativeConfigError, match="messages"):
        validate_config(config, "bad-agent")


def test_validate_call_llm_tools_without_tool_node():
    yaml_content = """\
manifest_version: 1

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            tools:
              - id: some-tool
            state_updates:
              messages: '[llm_response]'
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
manifest_version: 1

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
manifest_version: 1

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
manifest_version: 1

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
manifest_version: 1

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
manifest_version: 1

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            use_tools_from_registry: true
            state_updates:
              messages: '[llm_response]'
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
manifest_version: 1

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            use_tools_from_loaded_skills: true
            state_updates:
              messages: '[llm_response]'
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
    """use_tools_from_registry + explicit tools on call_llm is allowed."""
    yaml_content = """\
manifest_version: 1

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
      entry_point: agent
      nodes:
        - name: agent
          type: call_llm
          args:
            llm:
              id: gpt-4
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            use_tools_from_registry: true
            tools:
              - id: some-tool
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    validate_config(config, "agent")  # should not raise


def test_validate_loaded_skills_with_explicit_tools():
    """use_tools_from_loaded_skills + explicit tools on call_llm is allowed."""
    yaml_content = """\
manifest_version: 1

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
      entry_point: agent
      nodes:
        - name: agent
          type: call_llm
          args:
            llm:
              id: gpt-4
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            use_tools_from_loaded_skills: true
            tools:
              - id: some-tool
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    validate_config(config, "agent")  # should not raise


def test_validate_both_registry_and_loaded_skills():
    """use_tools_from_registry + use_tools_from_loaded_skills is an error."""
    yaml_content = """\
manifest_version: 1

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            use_tools_from_registry: true
            use_tools_from_loaded_skills: true
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    with pytest.raises(DeclarativeConfigError, match="cannot specify more than one"):
        validate_config(config, "bad-agent")


def test_validate_response_format_with_tools_raises():
    """response_format + tools on call_llm is an error."""
    yaml_content = """\
manifest_version: 1

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            response_format:
              name: UserInfo
              schema:
                type: object
                properties:
                  name:
                    type: string
            tools:
              - id: some-tool
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    with pytest.raises(DeclarativeConfigError, match="cannot use both"):
        validate_config(config, "bad-agent")


def test_validate_response_format_with_registry_tools_raises():
    """response_format + use_tools_from_registry on call_llm is an error."""
    yaml_content = """\
manifest_version: 1

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            response_format:
              name: UserInfo
              schema:
                type: object
                properties:
                  name:
                    type: string
            use_tools_from_registry: true
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    with pytest.raises(DeclarativeConfigError, match="cannot use both"):
        validate_config(config, "bad-agent")


def test_validate_response_format_without_tools_ok():
    """response_format alone on call_llm is valid."""
    yaml_content = """\
manifest_version: 1

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
      entry_point: agent
      nodes:
        - name: agent
          type: call_llm
          args:
            llm:
              id: gpt-4
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            response_format:
              name: UserInfo
              schema:
                type: object
                properties:
                  name:
                    type: string
            state_updates:
              messages: '[llm_response]'
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    validate_config(config, "agent")  # should not raise


def test_validate_tools_without_tool_node():
    """call_llm with tool options requires a tool_node in the graph."""
    yaml_content = """\
manifest_version: 1

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            use_tools_from_registry: true
            state_updates:
              messages: '[llm_response]'
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    with pytest.raises(DeclarativeConfigError, match="tool_node"):
        validate_config(config, "bad-agent")


@pytest.mark.asyncio
async def test_base_path_resolves_relative_skill_card_path(tmp_path):
    """Relative skill_card_path is resolved against base_path."""
    import json

    card_data = {
        "name": "Test Skill",
        "description": "A test skill",
        "base_uri": str(tmp_path),
        "files": ["SKILL.md"],
    }
    card_file = tmp_path / "skill-card.json"
    card_file.write_text(json.dumps(card_data))

    yaml_content = """\
manifest_version: 1

skills:
  - id: test-skill
    version: "1.0.0"
    skill_card_path: "skill-card.json"

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
    await populate_registries(config, registries, base_path=tmp_path)

    skill = await registries.skill_registry.get("test-skill", "==1.0.0")
    assert skill.skill_card is not None
    assert skill.skill_card.name == "Test Skill"


@pytest.mark.asyncio
async def test_base_path_resolves_relative_sub_agent_yaml_path(tmp_path):
    """Relative sub-agent yaml_path is resolved against base_path."""
    from unittest.mock import MagicMock

    sub_yaml = tmp_path / "sub.yaml"
    sub_yaml.write_text("""\
manifest_version: 1

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

agents:
  sub-agent:
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
              x: '"hi"'
      edges: []
""")

    yaml_content = """\
manifest_version: 1

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

sub_agents:
  - id: sub-agent
    version: "1.0.0"
    yaml_path: "sub.yaml"

agents:
  main:
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
              x: '"hi"'
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    registries = RegistryBundle(chat_models={"gpt-4": MagicMock()})
    await populate_registries(config, registries, base_path=tmp_path)

    agent = await registries.agent_registry.get("sub-agent", "==1.0.0")
    assert agent is not None


@pytest.mark.asyncio
async def test_relative_skill_card_path_without_base_path_raises():
    """Relative skill_card_path without base_path raises DeclarativeConfigError."""
    yaml_content = """\
manifest_version: 1

skills:
  - id: test-skill
    version: "1.0.0"
    skill_card_path: "relative/path/card.json"

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
    with pytest.raises(DeclarativeConfigError, match="requires a base_path"):
        await populate_registries(config, registries)


@pytest.mark.asyncio
async def test_relative_sub_agent_yaml_path_without_base_path_raises():
    """Relative sub-agent yaml_path without base_path raises DeclarativeConfigError."""
    yaml_content = """\
manifest_version: 1

sub_agents:
  - id: sub-agent
    version: "1.0.0"
    yaml_path: "relative/sub.yaml"

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
    with pytest.raises(DeclarativeConfigError, match="requires a base_path"):
        await populate_registries(config, registries)


@pytest.mark.asyncio
async def test_absolute_skill_card_path_ignores_base_path(tmp_path):
    """Absolute skill_card_path works regardless of base_path."""
    import json

    card_data = {
        "name": "Abs Skill",
        "description": "An absolute skill",
        "base_uri": str(tmp_path),
        "files": ["SKILL.md"],
    }
    card_file = tmp_path / "skill-card.json"
    card_file.write_text(json.dumps(card_data))

    yaml_content = f"""\
manifest_version: 1

skills:
  - id: abs-skill
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
    # Pass a different base_path — should be ignored for absolute path
    await populate_registries(config, registries, base_path=Path("/some/other/path"))

    skill = await registries.skill_registry.get("abs-skill", "==1.0.0")
    assert skill.skill_card is not None
    assert skill.skill_card.name == "Abs Skill"


def test_skill_def_with_skill_card_path():
    yaml_content = """\
manifest_version: 1

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


@pytest.mark.anyio
async def test_populate_registries_propagates_tenant_id():
    """All entries and entity instances should carry the bundle's tenant_id."""
    from unittest.mock import MagicMock

    config = load_declarative_config(yaml_content=MINIMAL_YAML)
    registries = RegistryBundle(
        tenant_id="acme",
        chat_models={"gpt-4": MagicMock()},
    )
    await populate_registries(config, registries)

    llm = await registries.llm_registry.get("gpt-4")
    assert llm.tenant_id == "acme"

    prompt = await registries.prompt_registry.get("sys")
    assert prompt.tenant_id == "acme"


# ---------------------------------------------------------------------------
# on_chat_model_create hook tests
# ---------------------------------------------------------------------------

HOOK_YAML = """\
manifest_version: 1

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

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


@pytest.mark.asyncio
async def test_on_chat_model_create_modifies_kwargs():
    """Hook that modifies kwargs (e.g., adds temperature) is applied."""
    from unittest.mock import MagicMock, patch

    from sherma.hooks.executor import BaseHookExecutor
    from sherma.hooks.manager import HookManager
    from sherma.hooks.types import ChatModelCreateContext

    class TempHook(BaseHookExecutor):
        async def on_chat_model_create(
            self, ctx: ChatModelCreateContext
        ) -> ChatModelCreateContext:
            ctx.kwargs["temperature"] = 0.5
            return ctx

    hook_manager = HookManager()
    hook_manager.register(TempHook())

    config = load_declarative_config(yaml_content=HOOK_YAML)
    registries = RegistryBundle()

    mock_chat = MagicMock()
    with patch(
        "sherma.langgraph.declarative.loader._construct_chat_model",
        return_value=mock_chat,
    ) as mock_construct:
        await populate_registries(config, registries, hook_manager=hook_manager)

    assert registries.chat_models["gpt-4"] is mock_chat
    # Verify temperature was passed through kwargs
    call_kwargs = mock_construct.call_args[0][1]
    assert call_kwargs["temperature"] == 0.5


@pytest.mark.asyncio
async def test_on_chat_model_create_provides_custom_model():
    """Hook that sets chat_model bypasses default construction."""
    from unittest.mock import MagicMock, patch

    from sherma.hooks.executor import BaseHookExecutor
    from sherma.hooks.manager import HookManager
    from sherma.hooks.types import ChatModelCreateContext

    custom_model = MagicMock(name="custom_model")

    class CustomModelHook(BaseHookExecutor):
        async def on_chat_model_create(
            self, ctx: ChatModelCreateContext
        ) -> ChatModelCreateContext:
            ctx.chat_model = custom_model
            return ctx

    hook_manager = HookManager()
    hook_manager.register(CustomModelHook())

    config = load_declarative_config(yaml_content=HOOK_YAML)
    registries = RegistryBundle()

    with patch(
        "sherma.langgraph.declarative.loader._construct_chat_model",
    ) as mock_construct:
        await populate_registries(config, registries, hook_manager=hook_manager)

    assert registries.chat_models["gpt-4"] is custom_model
    mock_construct.assert_not_called()


@pytest.mark.asyncio
async def test_no_hook_manager_default_behavior():
    """Without a hook manager, default chat model creation is used."""
    from unittest.mock import MagicMock, patch

    config = load_declarative_config(yaml_content=HOOK_YAML)
    registries = RegistryBundle()

    mock_chat = MagicMock()
    with patch(
        "sherma.langgraph.declarative.loader._construct_chat_model",
        return_value=mock_chat,
    ):
        await populate_registries(config, registries)

    assert registries.chat_models["gpt-4"] is mock_chat


# ---------------------------------------------------------------------------
# populate_hooks with remote URL tests
# ---------------------------------------------------------------------------


def test_populate_hooks_with_url():
    """populate_hooks creates a RemoteHookExecutor for url-based hooks."""
    yaml_content = """\
manifest_version: 1

hooks:
  - url: "http://localhost:8080/hooks"

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
    hook_manager = HookManager()
    populate_hooks(config, hook_manager)

    assert len(hook_manager._executors) == 1
    assert isinstance(hook_manager._executors[0], RemoteHookExecutor)
    assert hook_manager._executors[0]._url == "http://localhost:8080/hooks"


def test_populate_hooks_mixed():
    """populate_hooks handles a mix of url and import_path hooks."""
    yaml_content = """\
manifest_version: 1

hooks:
  - url: "http://localhost:8080/hooks"
  - import_path: "sherma.hooks.executor.BaseHookExecutor"

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
    hook_manager = HookManager()
    populate_hooks(config, hook_manager)

    assert len(hook_manager._executors) == 2
    assert isinstance(hook_manager._executors[0], RemoteHookExecutor)
    from sherma.hooks.executor import BaseHookExecutor

    assert isinstance(hook_manager._executors[1], BaseHookExecutor)


# ---------------------------------------------------------------------------
# use_sub_agents_as_tools validation tests
# ---------------------------------------------------------------------------


def test_validate_use_sub_agents_as_tools_all():
    """use_sub_agents_as_tools: true (parsed as 'all') with sub_agents is valid."""
    yaml_content = """\
manifest_version: 1

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

sub_agents:
  - id: weather-agent
    version: "1.0.0"
    url: "http://localhost:8080"

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            use_sub_agents_as_tools: true
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    from sherma.langgraph.declarative.schema import CallLLMArgs

    llm_node = config.agents["agent"].graph.nodes[0]
    assert isinstance(llm_node.args, CallLLMArgs)
    assert llm_node.args.use_sub_agents_as_tools == "all"
    validate_config(config, "agent")  # should not raise


def test_validate_use_sub_agents_as_tools_explicit_all():
    """use_sub_agents_as_tools: all (string) with sub_agents is valid."""
    yaml_content = """\
manifest_version: 1

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

sub_agents:
  - id: weather-agent
    version: "1.0.0"
    url: "http://localhost:8080"

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            use_sub_agents_as_tools: all
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    from sherma.langgraph.declarative.schema import CallLLMArgs

    llm_node = config.agents["agent"].graph.nodes[0]
    assert isinstance(llm_node.args, CallLLMArgs)
    assert llm_node.args.use_sub_agents_as_tools == "all"
    validate_config(config, "agent")  # should not raise


def test_validate_use_sub_agents_as_tools_list():
    """use_sub_agents_as_tools with list of refs is valid when IDs exist."""
    yaml_content = """\
manifest_version: 1

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

sub_agents:
  - id: weather-agent
    version: "1.0.0"
    url: "http://localhost:8080"
  - id: search-agent
    version: "1.0.0"
    url: "http://localhost:8081"

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            use_sub_agents_as_tools:
              - id: weather-agent
                version: "1.0.0"
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    from sherma.langgraph.declarative.schema import CallLLMArgs

    llm_node = config.agents["agent"].graph.nodes[0]
    assert isinstance(llm_node.args, CallLLMArgs)
    assert isinstance(llm_node.args.use_sub_agents_as_tools, list)
    assert len(llm_node.args.use_sub_agents_as_tools) == 1
    validate_config(config, "agent")  # should not raise


def test_validate_use_sub_agents_as_tools_list_unknown_id():
    """use_sub_agents_as_tools with unknown sub-agent ID raises error."""
    yaml_content = """\
manifest_version: 1

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

sub_agents:
  - id: weather-agent
    version: "1.0.0"
    url: "http://localhost:8080"

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
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            use_sub_agents_as_tools:
              - id: nonexistent-agent
                version: "1.0.0"
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    with pytest.raises(DeclarativeConfigError, match="nonexistent-agent"):
        validate_config(config, "agent")


@pytest.mark.asyncio
async def test_populate_prompt_instructions_path_relative(tmp_path):
    """Relative instructions_path is resolved against base_path."""
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "sys.md").write_text("You are a test prompt.")
    yaml_content = """\
manifest_version: 1

prompts:
  - id: sys
    version: "1.0.0"
    instructions_path: "prompts/sys.md"

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
    await populate_registries(config, registries, base_path=tmp_path)

    prompt = await registries.prompt_registry.get("sys", "==1.0.0")
    assert prompt.instructions == "You are a test prompt."


@pytest.mark.asyncio
async def test_populate_prompt_instructions_path_absolute(tmp_path):
    """Absolute instructions_path is loaded without base_path."""
    prompt_file = tmp_path / "abs.md"
    prompt_file.write_text("Absolute prompt body.")
    yaml_content = f"""\
manifest_version: 1

prompts:
  - id: sys
    version: "1.0.0"
    instructions_path: "{prompt_file}"

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

    prompt = await registries.prompt_registry.get("sys", "==1.0.0")
    assert prompt.instructions == "Absolute prompt body."


@pytest.mark.asyncio
async def test_populate_prompt_instructions_path_missing_file(tmp_path):
    """Missing instructions_path file raises DeclarativeConfigError."""
    yaml_content = """\
manifest_version: 1

prompts:
  - id: sys
    version: "1.0.0"
    instructions_path: "prompts/missing.md"

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
    with pytest.raises(DeclarativeConfigError, match="instructions file not found"):
        await populate_registries(config, registries, base_path=tmp_path)


@pytest.mark.asyncio
async def test_populate_prompt_relative_instructions_path_without_base_path():
    """Relative instructions_path without base_path raises DeclarativeConfigError."""
    yaml_content = """\
manifest_version: 1

prompts:
  - id: sys
    version: "1.0.0"
    instructions_path: "prompts/sys.md"

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
    with pytest.raises(DeclarativeConfigError, match="requires a base_path"):
        await populate_registries(config, registries)


def test_prompt_def_requires_instructions_or_path():
    """PromptDef with neither instructions nor instructions_path raises."""
    yaml_content = """\
manifest_version: 1

prompts:
  - id: sys
    version: "1.0.0"

agents: {}
"""
    with pytest.raises(Exception, match="instructions"):
        load_declarative_config(yaml_content=yaml_content)


def test_prompt_def_rejects_both_instructions_and_path():
    """PromptDef with both instructions and instructions_path raises."""
    yaml_content = """\
manifest_version: 1

prompts:
  - id: sys
    version: "1.0.0"
    instructions: "inline"
    instructions_path: "file.md"

agents: {}
"""
    with pytest.raises(Exception, match="cannot have both"):
        load_declarative_config(yaml_content=yaml_content)


# -- build_checkpointer tests -------------------------------------------


class _FakeAsyncSaver:
    """Minimal fake async checkpointer.

    Acts as both the ``from_conn_string`` async context manager and the
    saver instance yielded from it.  Records the URL it was opened with
    and whether ``asetup``/``aclose`` were invoked.
    """

    opened_urls: ClassVar[list[str]] = []
    entered_count: ClassVar[int] = 0
    exited_count: ClassVar[int] = 0

    def __init__(self, url: str) -> None:
        self.url = url
        self.asetup_calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.opened_urls = []
        cls.entered_count = 0
        cls.exited_count = 0

    @classmethod
    def from_conn_string(cls, url: str) -> _FakeAsyncSaver:
        cls.opened_urls.append(url)
        return cls(url)

    async def __aenter__(self) -> _FakeAsyncSaver:
        type(self).entered_count += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        type(self).exited_count += 1

    async def asetup(self) -> None:
        self.asetup_calls += 1


@pytest.mark.asyncio
async def test_build_checkpointer_none_returns_none_none():
    saver, stack = await build_checkpointer(None)
    assert saver is None
    assert stack is None


@pytest.mark.asyncio
async def test_build_checkpointer_memory_returns_memory_saver():
    from langgraph.checkpoint.memory import MemorySaver

    saver, stack = await build_checkpointer(MemoryCheckpointerDef())
    assert isinstance(saver, MemorySaver)
    assert stack is None


@pytest.mark.asyncio
async def test_build_checkpointer_redis(monkeypatch: pytest.MonkeyPatch):
    _FakeAsyncSaver.reset()
    from sherma.langgraph.declarative import loader

    monkeypatch.setattr(loader, "_import_redis_saver", lambda: _FakeAsyncSaver)

    defn = RedisCheckpointerDef(type="redis", url="redis://localhost:6379")
    saver, stack = await build_checkpointer(defn)

    assert _FakeAsyncSaver.opened_urls == ["redis://localhost:6379"]
    assert _FakeAsyncSaver.entered_count == 1
    assert _FakeAsyncSaver.exited_count == 0
    assert isinstance(saver, _FakeAsyncSaver)
    assert saver.asetup_calls == 1
    assert stack is not None

    await stack.aclose()
    assert _FakeAsyncSaver.exited_count == 1


@pytest.mark.asyncio
async def test_build_checkpointer_postgres(monkeypatch: pytest.MonkeyPatch):
    _FakeAsyncSaver.reset()
    from sherma.langgraph.declarative import loader

    monkeypatch.setattr(loader, "_import_postgres_saver", lambda: _FakeAsyncSaver)

    defn = PostgresCheckpointerDef(
        type="postgres",
        url="postgresql://user@localhost:5432/app",
    )
    saver, stack = await build_checkpointer(defn)

    assert _FakeAsyncSaver.opened_urls == ["postgresql://user@localhost:5432/app"]
    assert isinstance(saver, _FakeAsyncSaver)
    assert saver.asetup_calls == 1
    assert stack is not None
    await stack.aclose()
    assert _FakeAsyncSaver.exited_count == 1


@pytest.mark.asyncio
async def test_build_checkpointer_redis_missing_package(
    monkeypatch: pytest.MonkeyPatch,
):
    from sherma.langgraph.declarative import loader

    def _raise() -> type:
        raise DeclarativeConfigError(
            "Redis checkpointer requires the 'sherma[redis]' extra."
        )

    monkeypatch.setattr(loader, "_import_redis_saver", _raise)

    defn = RedisCheckpointerDef(type="redis", url="redis://localhost:6379")
    with pytest.raises(DeclarativeConfigError, match="sherma\\[redis\\]"):
        await build_checkpointer(defn)


@pytest.mark.asyncio
async def test_build_checkpointer_postgres_missing_package(
    monkeypatch: pytest.MonkeyPatch,
):
    from sherma.langgraph.declarative import loader

    def _raise() -> type:
        raise DeclarativeConfigError(
            "Postgres checkpointer requires the 'sherma[postgres]' extra."
        )

    monkeypatch.setattr(loader, "_import_postgres_saver", _raise)

    defn = PostgresCheckpointerDef(
        type="postgres", url="postgresql://user@host:5432/db"
    )
    with pytest.raises(DeclarativeConfigError, match="sherma\\[postgres\\]"):
        await build_checkpointer(defn)


@pytest.mark.asyncio
async def test_build_checkpointer_hook_returns_instance():
    """Hook returning a pre-built saver short-circuits the builder."""
    from langgraph.checkpoint.memory import MemorySaver

    custom_saver = MemorySaver()

    class _HookExec(BaseHookExecutor):
        async def on_checkpointer_create(
            self, ctx: CheckpointerCreateContext
        ) -> CheckpointerCreateContext | None:
            ctx.checkpointer = custom_saver
            return ctx

    hm = HookManager()
    hm.register(_HookExec())

    defn = RedisCheckpointerDef(type="redis", url="redis://localhost:6379")
    saver, stack = await build_checkpointer(defn, hook_manager=hm)

    # Hook returned saver is used; default redis path was never invoked.
    assert saver is custom_saver
    assert stack is None


@pytest.mark.asyncio
async def test_build_checkpointer_hook_rewrites_definition(
    monkeypatch: pytest.MonkeyPatch,
):
    """Hook leaving ``checkpointer=None`` falls through to default builder."""
    _FakeAsyncSaver.reset()
    from sherma.langgraph.declarative import loader

    monkeypatch.setattr(loader, "_import_redis_saver", lambda: _FakeAsyncSaver)

    class _HookExec(BaseHookExecutor):
        async def on_checkpointer_create(
            self, ctx: CheckpointerCreateContext
        ) -> CheckpointerCreateContext | None:
            # Rewrite the definition to a different URL.
            ctx.definition = RedisCheckpointerDef(
                type="redis", url="redis://rewritten:6379"
            )
            return ctx

    hm = HookManager()
    hm.register(_HookExec())

    defn = RedisCheckpointerDef(type="redis", url="redis://localhost:6379")
    saver, stack = await build_checkpointer(defn, hook_manager=hm)

    assert _FakeAsyncSaver.opened_urls == ["redis://rewritten:6379"]
    assert isinstance(saver, _FakeAsyncSaver)
    assert stack is not None
    await stack.aclose()


@pytest.mark.asyncio
async def test_build_checkpointer_hook_can_install_from_none():
    """Hook can install a saver even when YAML has no checkpointer block."""
    from langgraph.checkpoint.memory import MemorySaver

    custom_saver = MemorySaver()

    class _HookExec(BaseHookExecutor):
        async def on_checkpointer_create(
            self, ctx: CheckpointerCreateContext
        ) -> CheckpointerCreateContext | None:
            assert ctx.definition is None
            ctx.checkpointer = custom_saver
            return ctx

    hm = HookManager()
    hm.register(_HookExec())

    saver, stack = await build_checkpointer(None, hook_manager=hm)
    assert saver is custom_saver
    assert stack is None
