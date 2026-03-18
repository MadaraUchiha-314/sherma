"""Tests for declarative config transform (auto-inject tool_nodes)."""

from __future__ import annotations

import pytest

from sherma.exceptions import DeclarativeConfigError
from sherma.langgraph.declarative.loader import load_declarative_config
from sherma.langgraph.declarative.schema import ToolNodeArgs
from sherma.langgraph.declarative.transform import inject_tool_nodes


def _base_yaml(nodes_yaml: str, edges_yaml: str = "edges: []") -> str:
    return f"""\
manifest_version: 1

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

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
{nodes_yaml}
      {edges_yaml}
"""


class TestInjectToolNodesExplicitTools:
    """Injection with explicit tools list."""

    def test_injects_tool_node_and_edges(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'messages'
            tools:
              - id: some-tool""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        result = inject_tool_nodes(config)

        agent = result.agents["test-agent"]
        node_names = {n.name for n in agent.graph.nodes}
        assert "agent_tools" in node_names

        # tool_node should have default args
        tool_node = next(n for n in agent.graph.nodes if n.name == "agent_tools")
        assert tool_node.type == "tool_node"
        assert isinstance(tool_node.args, ToolNodeArgs)

        # Should have conditional edge from agent
        cond_edge = next(
            e for e in agent.graph.edges if e.source == "agent" and e.branches
        )
        assert cond_edge.branches is not None
        assert len(cond_edge.branches) == 1
        assert cond_edge.branches[0].condition == "has_tool_calls"
        assert cond_edge.branches[0].target == "agent_tools"
        assert cond_edge.default == "__end__"

        # Should have loop-back edge
        loop_edge = next(
            e
            for e in agent.graph.edges
            if e.source == "agent_tools" and e.target == "agent"
        )
        assert loop_edge is not None


class TestInjectToolNodesRegistryTools:
    """Injection with use_tools_from_registry."""

    def test_injects_for_registry_tools(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'messages'
            use_tools_from_registry: true""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        result = inject_tool_nodes(config)

        agent = result.agents["test-agent"]
        node_names = {n.name for n in agent.graph.nodes}
        assert "agent_tools" in node_names


class TestInjectToolNodesLoadedSkills:
    """Injection with use_tools_from_loaded_skills."""

    def test_injects_for_loaded_skills(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'messages'
            use_tools_from_loaded_skills: true""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        result = inject_tool_nodes(config)

        agent = result.agents["test-agent"]
        node_names = {n.name for n in agent.graph.nodes}
        assert "agent_tools" in node_names


class TestBackwardCompat:
    """Existing conditional edges prevent injection."""

    def test_existing_has_tool_calls_skipped(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'messages'
            tools:
              - id: some-tool
        - name: my_tools
          type: tool_node
          args: {}""",
            """\
edges:
        - source: agent
          branches:
            - condition: has_tool_calls
              target: my_tools
          default: __end__
        - source: my_tools
          target: agent""",
        )
        config = load_declarative_config(yaml_content=yaml)
        result = inject_tool_nodes(config)

        agent = result.agents["test-agent"]
        node_names = [n.name for n in agent.graph.nodes]
        # No new tool node injected
        assert "agent_tools" not in node_names
        # Original nodes preserved
        assert "my_tools" in node_names

    def test_existing_custom_conditional_skipped(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'messages'
            tools:
              - id: some-tool
        - name: other
          type: set_state
          args:
            values:
              x: '"done"'""",
            """\
edges:
        - source: agent
          branches:
            - condition: 'size(messages) > 5'
              target: other
          default: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        result = inject_tool_nodes(config)

        agent = result.agents["test-agent"]
        node_names = [n.name for n in agent.graph.nodes]
        assert "agent_tools" not in node_names


class TestNoTools:
    """call_llm without tools gets no injection."""

    def test_no_injection_without_tools(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'messages'""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        result = inject_tool_nodes(config)

        agent = result.agents["test-agent"]
        assert len(agent.graph.nodes) == 1
        assert len(agent.graph.edges) == 1


class TestMultipleCallLLM:
    """Multiple call_llm nodes each get their own tool_node."""

    def test_multiple_injections(self):
        yaml = """\
manifest_version: 1

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

agents:
  test-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
    graph:
      entry_point: first
      nodes:
        - name: first
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'messages'
            tools:
              - id: tool-a
        - name: second
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"world"'
              - role: messages
                content: 'messages'
            tools:
              - id: tool-b
      edges:
        - source: first
          target: second
        - source: second
          target: __end__
"""
        config = load_declarative_config(yaml_content=yaml)
        result = inject_tool_nodes(config)

        agent = result.agents["test-agent"]
        node_names = {n.name for n in agent.graph.nodes}
        assert "first_tools" in node_names
        assert "second_tools" in node_names

        # first's conditional should default to second
        cond_first = next(
            e for e in agent.graph.edges if e.source == "first" and e.branches
        )
        assert cond_first.default == "second"

        # second's conditional should default to __end__
        cond_second = next(
            e for e in agent.graph.edges if e.source == "second" and e.branches
        )
        assert cond_second.default == "__end__"


class TestNameCollision:
    """Name collision raises DeclarativeConfigError."""

    def test_collision_raises(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'messages'
            tools:
              - id: some-tool
        - name: agent_tools
          type: set_state
          args:
            values:
              x: '"hi"'""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        with pytest.raises(
            DeclarativeConfigError,
            match=r"agent_tools.*already exists",
        ):
            inject_tool_nodes(config)


class TestNoOutgoingEdge:
    """No outgoing edge defaults to __end__."""

    def test_defaults_to_end(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'messages'
            tools:
              - id: some-tool""",
        )
        config = load_declarative_config(yaml_content=yaml)
        result = inject_tool_nodes(config)

        agent = result.agents["test-agent"]
        cond_edge = next(
            e for e in agent.graph.edges if e.source == "agent" and e.branches
        )
        assert cond_edge.default == "__end__"


class TestEdgeTargetPreserved:
    """Simple edge target is preserved as conditional default."""

    def test_target_becomes_default(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'messages'
            tools:
              - id: some-tool
        - name: next_step
          type: set_state
          args:
            values:
              x: '"done"'""",
            """\
edges:
        - source: agent
          target: next_step""",
        )
        config = load_declarative_config(yaml_content=yaml)
        result = inject_tool_nodes(config)

        agent = result.agents["test-agent"]
        cond_edge = next(
            e for e in agent.graph.edges if e.source == "agent" and e.branches
        )
        assert cond_edge.default == "next_step"

        # Original simple edge should be removed
        simple_edges = [
            e
            for e in agent.graph.edges
            if e.source == "agent" and e.target is not None and e.branches is None
        ]
        assert len(simple_edges) == 0


class TestOriginalConfigUnmodified:
    """inject_tool_nodes should not mutate the input config."""

    def test_deep_copy(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'messages'
            tools:
              - id: some-tool""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        original_node_count = len(config.agents["test-agent"].graph.nodes)

        inject_tool_nodes(config)

        # Original should be untouched
        assert len(config.agents["test-agent"].graph.nodes) == original_node_count
