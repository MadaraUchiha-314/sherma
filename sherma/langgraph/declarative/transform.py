"""Config transformations applied after parsing, before validation."""

from __future__ import annotations

import copy

from sherma.exceptions import DeclarativeConfigError
from sherma.langgraph.declarative.schema import (
    BranchDef,
    CallLLMArgs,
    DeclarativeConfig,
    EdgeDef,
    NodeDef,
    ToolNodeArgs,
)


def _has_tools(args: CallLLMArgs) -> bool:
    """Return True if the call_llm node has any tool binding."""
    return bool(
        args.tools
        or args.use_tools_from_registry
        or args.use_tools_from_loaded_skills
        or args.use_sub_agents_as_tools
    )


def inject_tool_nodes(config: DeclarativeConfig) -> DeclarativeConfig:
    """Auto-inject tool_node + conditional edges for call_llm nodes with tools.

    For each ``call_llm`` node that has tools bound but no existing conditional
    edge routing, this transform:

    1. Creates a ``tool_node`` named ``{name}_tools``
    2. Replaces the simple outgoing edge with a conditional edge
       (``has_tool_calls`` → tool_node, default → original target)
    3. Adds a loop-back edge from the tool_node back to the call_llm node

    Nodes that already have a conditional edge from them are left untouched
    (backward compatibility).
    """
    config = copy.deepcopy(config)

    for agent_name, agent_def in config.agents.items():
        graph = agent_def.graph
        node_names = {n.name for n in graph.nodes}

        # Index: which nodes already have conditional edges from them
        nodes_with_conditional = {
            e.source for e in graph.edges if e.branches is not None
        }

        new_nodes: list[NodeDef] = []
        new_edges: list[EdgeDef] = []
        edges_to_remove: set[int] = set()

        for node_def in graph.nodes:
            if node_def.type != "call_llm":
                continue

            args: CallLLMArgs = node_def.args  # type: ignore[assignment]
            if not _has_tools(args):
                continue

            # Skip if there's already a conditional edge from this node
            if node_def.name in nodes_with_conditional:
                continue

            tool_node_name = f"{node_def.name}_tools"

            # Check for name collision
            if tool_node_name in node_names:
                raise DeclarativeConfigError(
                    f"Cannot auto-inject tool_node: name '{tool_node_name}' "
                    f"already exists in agent '{agent_name}'"
                )

            # Find the existing simple edge from this node
            default_target = "__end__"
            for idx, edge in enumerate(graph.edges):
                if edge.source == node_def.name and edge.target is not None:
                    default_target = edge.target
                    edges_to_remove.add(idx)
                    break

            # Create tool_node
            new_nodes.append(
                NodeDef(
                    name=tool_node_name,
                    type="tool_node",
                    args=ToolNodeArgs(),
                )
            )
            node_names.add(tool_node_name)

            # Conditional: call_llm → has_tool_calls → tool_node
            new_edges.append(
                EdgeDef(
                    source=node_def.name,
                    branches=[
                        BranchDef(
                            condition="has_tool_calls",
                            target=tool_node_name,
                        )
                    ],
                    default=default_target,
                )
            )

            # Create loop-back edge: tool_node → call_llm
            new_edges.append(
                EdgeDef(
                    source=tool_node_name,
                    target=node_def.name,
                )
            )

        # Apply modifications
        if new_nodes or edges_to_remove:
            graph.nodes.extend(new_nodes)
            graph.edges = [
                e for idx, e in enumerate(graph.edges) if idx not in edges_to_remove
            ] + new_edges

    return config
