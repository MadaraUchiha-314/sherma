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

# Built-in condition key recognized by the edge router for fallback routing.
HAS_ERROR_FALLBACK = "has_error_fallback"


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


def inject_fallback_edges(config: DeclarativeConfig) -> DeclarativeConfig:
    """Auto-inject conditional edges for nodes with ``on_error.fallback``.

    For each node that declares ``on_error.fallback``, this transform
    replaces its outgoing simple edge with a conditional edge:

    * ``has_error_fallback`` → fallback node
    * default → original target

    If the node already has a conditional edge, the fallback branch is
    prepended to the existing branches so it takes priority.
    """
    config = copy.deepcopy(config)

    for _agent_name, agent_def in config.agents.items():
        graph = agent_def.graph
        node_names = {n.name for n in graph.nodes}

        edges_to_remove: set[int] = set()
        new_edges: list[EdgeDef] = []

        # Collect nodes that have on_error.fallback
        fallback_nodes: dict[str, str] = {}  # node_name → fallback target
        for node_def in graph.nodes:
            if node_def.on_error and node_def.on_error.fallback:
                fallback_target = node_def.on_error.fallback
                if fallback_target not in node_names:
                    raise DeclarativeConfigError(
                        f"on_error.fallback target '{fallback_target}' "
                        f"for node '{node_def.name}' does not exist"
                    )
                fallback_nodes[node_def.name] = fallback_target

        if not fallback_nodes:
            continue

        for idx, edge in enumerate(graph.edges):
            if edge.source not in fallback_nodes:
                continue

            fallback_target = fallback_nodes[edge.source]
            fallback_branch = BranchDef(
                condition=HAS_ERROR_FALLBACK,
                target=fallback_target,
            )

            if edge.branches is not None:
                # Existing conditional edge — prepend fallback branch
                edge.branches.insert(0, fallback_branch)
            else:
                # Simple edge — replace with conditional
                edges_to_remove.add(idx)
                new_edges.append(
                    EdgeDef(
                        source=edge.source,
                        branches=[fallback_branch],
                        default=edge.target or "__end__",
                    )
                )

            # Remove from fallback_nodes so we don't process twice
            del fallback_nodes[edge.source]

        # Apply modifications
        if edges_to_remove or new_edges:
            graph.edges = [
                e for i, e in enumerate(graph.edges) if i not in edges_to_remove
            ] + new_edges

    return config
