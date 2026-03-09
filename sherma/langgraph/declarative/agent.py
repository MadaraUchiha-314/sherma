"""DeclarativeAgent: LangGraph agent defined entirely via YAML + CEL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ConfigDict

from sherma.exceptions import GraphConstructionError
from sherma.langgraph.agent import LangGraphAgent
from sherma.langgraph.declarative.cel_engine import CelEngine
from sherma.langgraph.declarative.edges import build_conditional_router
from sherma.langgraph.declarative.loader import (
    RegistryBundle,
    load_declarative_config,
    populate_registries,
    validate_config,
)
from sherma.langgraph.declarative.nodes import (
    build_call_agent_node,
    build_call_llm_node,
    build_data_transform_node,
    build_set_state_node,
    build_tool_node,
    resolve_tools_for_node_async,
)
from sherma.langgraph.declarative.schema import (
    CallAgentArgs,
    CallLLMArgs,
    DeclarativeConfig,
    NodeDef,
    ToolNodeArgs,
)
from sherma.logging import get_logger

logger = get_logger(__name__)

# Mapping from YAML type strings to Python types
_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}


def _build_state_class(
    agent_def: Any,
) -> type:
    """Build a dynamic TypedDict state class from the state schema."""
    from typing import TypedDict

    fields = agent_def.state.fields
    field_names = {f.name for f in fields}

    # If messages field exists, use MessagesState as base
    if "messages" in field_names:
        # Build annotations for extra fields
        extra_annotations: dict[str, Any] = {}
        for field_def in fields:
            if field_def.name == "messages":
                continue  # MessagesState already has this
            py_type = _TYPE_MAP.get(field_def.type, str)
            extra_annotations[field_def.name] = py_type

        if not extra_annotations:
            return MessagesState

        # Create a subclass of MessagesState with extra fields
        ns: dict[str, Any] = {"__annotations__": extra_annotations}
        return type("DynamicState", (MessagesState,), ns)

    # No messages field - build a TypedDict
    td_fields: dict[str, Any] = {}
    for field_def in fields:
        py_type = _TYPE_MAP.get(field_def.type, str)
        td_fields[field_def.name] = py_type

    return TypedDict("DynamicState", td_fields)  # type: ignore[call-overload]


class DeclarativeAgent(LangGraphAgent):
    """An agent defined declaratively via YAML and CEL expressions.

    Builds everything from the YAML config automatically:
    - LLMs are created from provider + model_name (API keys from env)
    - Tools are imported from import_path in the YAML
    - Prompts are loaded directly from the YAML

    No registries or chat model setup needed from user code.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    yaml_path: str | Path | None = None
    yaml_content: str | None = None
    http_async_client: Any | None = None
    _registries: RegistryBundle | None = None
    _compiled_graph: CompiledStateGraph | None = None

    async def get_graph(self) -> CompiledStateGraph:
        """Build and return the compiled LangGraph from YAML config."""
        if self._compiled_graph is not None:
            return self._compiled_graph

        # 1. Load and validate YAML
        config = load_declarative_config(
            yaml_path=self.yaml_path,
            yaml_content=self.yaml_content,
        )

        # Find the agent definition (use self.id to match)
        agent_name = self._find_agent_name(config)
        validate_config(config, agent_name)

        # 2. Auto-build registries from config
        if self._registries is None:
            self._registries = RegistryBundle()
        await populate_registries(config, self._registries, self.http_async_client)

        # 3. Build the graph
        agent_def = config.agents[agent_name]
        self._compiled_graph = await self._build_graph(agent_def, config)
        return self._compiled_graph

    def _find_agent_name(self, config: DeclarativeConfig) -> str:
        """Find the agent name in config matching this agent's id."""
        if self.id in config.agents:
            return self.id
        # If only one agent defined, use it
        if len(config.agents) == 1:
            return next(iter(config.agents))
        raise GraphConstructionError(
            f"Agent '{self.id}' not found in config. "
            f"Available: {list(config.agents.keys())}"
        )

    async def _build_graph(
        self,
        agent_def: Any,
        config: DeclarativeConfig,
    ) -> CompiledStateGraph:
        """Build a LangGraph StateGraph from the agent definition."""
        state_class = _build_state_class(agent_def)

        extra_vars = self._build_cel_extra_vars(config)
        cel = CelEngine(extra_vars=extra_vars)

        graph = StateGraph(state_class)

        # Add nodes
        all_nodes = list(agent_def.graph.nodes)
        for node_def in all_nodes:
            node_fn = await self._build_node(node_def, cel, all_nodes)
            graph.add_node(node_def.name, node_fn)

        # Add entry edge
        graph.add_edge(START, agent_def.graph.entry_point)

        # Add edges
        for edge_def in agent_def.graph.edges:
            if edge_def.branches:
                router, path_map = build_conditional_router(edge_def, cel)
                graph.add_conditional_edges(
                    edge_def.source,
                    router,
                    path_map,  # type: ignore[arg-type]
                )
            elif edge_def.target is not None:
                target = edge_def.target
                if target == "__end__":
                    from langgraph.graph import END

                    target = END
                graph.add_edge(edge_def.source, target)

        return graph.compile()

    async def _build_node(
        self,
        node_def: NodeDef,
        cel: CelEngine,
        all_nodes: list[NodeDef] | None = None,
    ) -> Any:
        """Build a node function from a node definition."""
        assert self._registries is not None

        if node_def.type == "call_llm":
            args: CallLLMArgs = node_def.args  # type: ignore[assignment]
            llm_id = args.llm.id
            if llm_id not in self._registries.chat_models:
                raise GraphConstructionError(
                    f"Chat model '{llm_id}' not found. "
                    f"Ensure llm '{llm_id}' is declared in the YAML "
                    f"with a valid provider."
                )
            chat_model = self._registries.chat_models[llm_id]
            tools = None
            if args.tools:
                tools = await resolve_tools_for_node_async(
                    args.tools, self._registries.tool_registry
                )
            return build_call_llm_node(node_def, chat_model, cel, tools)

        if node_def.type == "tool_node":
            tn_args: ToolNodeArgs = node_def.args  # type: ignore[assignment]
            tool_refs = tn_args.tools
            if not tool_refs:
                tool_refs = self._collect_llm_tools(all_nodes or [])
            if not tool_refs:
                raise GraphConstructionError(
                    f"tool_node '{node_def.name}' has no tools and no "
                    f"call_llm nodes with tools found to inherit from"
                )
            tools = await resolve_tools_for_node_async(
                tool_refs, self._registries.tool_registry
            )
            return build_tool_node(node_def, tools)

        if node_def.type == "call_agent":
            ca_args: CallAgentArgs = node_def.args  # type: ignore[assignment]
            agent = await self._registries.tool_registry.get(
                ca_args.agent.id, ca_args.agent.version
            )
            return build_call_agent_node(node_def, agent, cel)

        if node_def.type == "data_transform":
            return build_data_transform_node(node_def, cel)

        if node_def.type == "set_state":
            return build_set_state_node(node_def, cel)

        raise GraphConstructionError(f"Unknown node type: {node_def.type}")

    @staticmethod
    def _collect_llm_tools(
        all_nodes: list[NodeDef],
    ) -> list[Any]:
        """Collect all tool refs from call_llm nodes in the graph."""
        from sherma.langgraph.declarative.schema import RegistryRef

        tool_refs: list[RegistryRef] = []
        seen: set[tuple[str, str]] = set()
        for node in all_nodes:
            if node.type == "call_llm":
                llm_args: CallLLMArgs = node.args  # type: ignore[assignment]
                if llm_args.tools:
                    for ref in llm_args.tools:
                        key = (ref.id, ref.version)
                        if key not in seen:
                            seen.add(key)
                            tool_refs.append(ref)
        return tool_refs

    def _build_cel_extra_vars(self, config: DeclarativeConfig) -> dict[str, Any]:
        """Build extra variables for CEL from config registries."""
        extra: dict[str, Any] = {}

        prompts: dict[str, dict[str, str]] = {}
        for p in config.prompts:
            prompts[p.id] = {"instructions": p.instructions}
        if prompts:
            extra["prompts"] = prompts

        llms: dict[str, dict[str, str]] = {}
        for llm in config.llms:
            llms[llm.id] = {"model_name": llm.model_name}
        if llms:
            extra["llms"] = llms

        return extra
