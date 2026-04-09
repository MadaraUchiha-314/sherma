"""DeclarativeAgent: LangGraph agent defined entirely via YAML + CEL."""

from __future__ import annotations

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ConfigDict, Field, PrivateAttr

from sherma.entities.base import DEFAULT_TENANT_ID
from sherma.exceptions import GraphConstructionError
from sherma.hooks.executor import HookExecutor
from sherma.langgraph.agent import LangGraphAgent
from sherma.langgraph.declarative.cel_engine import CelEngine
from sherma.langgraph.declarative.edges import build_conditional_router
from sherma.langgraph.declarative.loader import (
    RegistryBundle,
    build_checkpointer,
    load_declarative_config,
    populate_hooks,
    populate_registries,
    validate_config,
)
from sherma.langgraph.declarative.nodes import (
    INTERNAL_STATE_KEY,
    NodeContext,
    build_call_agent_node,
    build_call_llm_node,
    build_custom_node,
    build_data_transform_node,
    build_interrupt_node,
    build_load_skills_node,
    build_set_state_node,
    build_tool_node,
)
from sherma.langgraph.declarative.schema import (
    CallAgentArgs,
    CallLLMArgs,
    DeclarativeConfig,
    NodeDef,
)
from sherma.langgraph.declarative.transform import (
    inject_fallback_edges,
    inject_tool_nodes,
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


def _needs_internal_state(agent_def: Any, *, has_skills: bool) -> bool:
    """Return ``True`` if the agent needs the ``__sherma__`` internal state key."""
    if has_skills:
        return True
    # Any node with on_error.fallback uses __sherma__ to store error info
    return any(n.on_error and n.on_error.fallback for n in agent_def.graph.nodes)


def _build_state_class(
    agent_def: Any,
    *,
    has_skills: bool = False,
) -> type:
    """Build a dynamic TypedDict state class from the state schema.

    The internal ``__sherma__`` field is auto-injected when the agent
    needs managed internal state (skills, error fallback routing, etc.).
    """
    from typing import TypedDict

    inject_internal = _needs_internal_state(agent_def, has_skills=has_skills)

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

        if inject_internal:
            extra_annotations[INTERNAL_STATE_KEY] = dict

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

    if inject_internal:
        td_fields[INTERNAL_STATE_KEY] = dict

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
    config: DeclarativeConfig | None = None
    base_path: Path | None = None
    http_async_client: Any | None = None
    hooks: list[HookExecutor] = Field(default_factory=list)
    tenant_id: str = DEFAULT_TENANT_ID
    checkpointer: BaseCheckpointSaver = Field(default_factory=MemorySaver)
    _registries: RegistryBundle | None = None
    _compiled_graph: CompiledStateGraph | None = None
    _checkpointer_exit_stack: AsyncExitStack | None = PrivateAttr(default=None)

    async def get_graph(self) -> CompiledStateGraph:
        """Build and return the compiled LangGraph from YAML config."""
        if self._compiled_graph is not None:
            return self._compiled_graph

        # 1. Load config from the provided source
        if self.config is not None:
            config = self.config
        else:
            config = load_declarative_config(
                yaml_path=self.yaml_path,
                yaml_content=self.yaml_content,
            )

        # Auto-inject tool_nodes for call_llm nodes with tools
        config = inject_tool_nodes(config)
        # Auto-inject fallback edges for nodes with on_error.fallback
        config = inject_fallback_edges(config)

        # Find the agent definition (use self.id to match)
        agent_name = self._find_agent_name(config)
        validate_config(config, agent_name)

        # Register hooks BEFORE populating registries (needed for on_chat_model_create)
        for executor in self.hooks:
            self.hook_manager.register(executor)

        if config.hooks:
            populate_hooks(config, self.hook_manager)

        # Derive base_path for resolving relative file paths in the YAML
        base_path = self.base_path
        if base_path is None and self.yaml_path is not None:
            base_path = Path(self.yaml_path).resolve().parent

        # 2. Auto-build registries from config
        if self._registries is None:
            self._registries = RegistryBundle(tenant_id=self.tenant_id)
        await populate_registries(
            config,
            self._registries,
            self.http_async_client,
            self.hook_manager,
            base_path=base_path,
        )

        # Track sub-agent tool IDs for use_sub_agents_as_tools
        self._sub_agent_tool_ids: list[str] = [sa.id for sa in config.sub_agents]

        # 3. Resolve checkpointer.  ``build_checkpointer`` runs the
        # ``on_checkpointer_create`` hook (if registered) first, then
        # falls back to instantiating from the YAML definition.  For
        # redis/postgres it opens connection resources through an
        # ``AsyncExitStack`` that is released in ``aclose``.
        saver, stack = await build_checkpointer(
            config.checkpointer, hook_manager=self.hook_manager
        )
        if saver is not None:
            checkpointer = saver
            self._checkpointer_exit_stack = stack
        else:
            checkpointer = self.checkpointer

        # 4. Apply langgraph_config from YAML to agent fields
        agent_def = config.agents[agent_name]
        if agent_def.langgraph_config:
            lgc = agent_def.langgraph_config
            if lgc.recursion_limit is not None:
                self.recursion_limit = lgc.recursion_limit
            if lgc.max_concurrency is not None:
                self.max_concurrency = lgc.max_concurrency
            if lgc.tags is not None:
                self.tags = lgc.tags
            if lgc.metadata is not None:
                self.metadata = lgc.metadata

        # 5. Build the graph
        self._compiled_graph = await self._build_graph(
            agent_def, config, checkpointer=checkpointer
        )
        return self._compiled_graph

    async def aclose(self) -> None:
        """Release resources owned by this agent.

        Closes the checkpointer connection pool (when redis/postgres
        was resolved through :func:`build_checkpointer`) and clears
        the cached compiled graph so the agent can be re-initialised.
        Safe to call multiple times.
        """
        if self._checkpointer_exit_stack is not None:
            try:
                await self._checkpointer_exit_stack.aclose()
            finally:
                self._checkpointer_exit_stack = None
        self._compiled_graph = None

    async def __aenter__(self) -> DeclarativeAgent:
        await self.get_graph()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

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
        *,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> CompiledStateGraph:
        """Build a LangGraph StateGraph from the agent definition."""
        state_class = _build_state_class(agent_def, has_skills=bool(config.skills))

        extra_vars = self._build_cel_extra_vars(config)
        cel = CelEngine(extra_vars=extra_vars)

        graph = StateGraph(state_class)

        # Add nodes
        has_skills = bool(config.skills)
        for node_def in agent_def.graph.nodes:
            node_fn = await self._build_node(
                node_def, cel, config, has_skills=has_skills
            )
            graph.add_node(node_def.name, node_fn)

        # Add entry edge (only when entry_point is explicitly set; when
        # edges with source "__start__" are used instead, they are wired
        # below in the edges loop).
        if agent_def.graph.entry_point is not None:
            graph.add_edge(START, agent_def.graph.entry_point)

        # Add edges
        from langgraph.graph import END

        for edge_def in agent_def.graph.edges:
            source: Any = START if edge_def.source == "__start__" else edge_def.source
            if edge_def.branches:
                router, path_map = build_conditional_router(edge_def, cel)
                graph.add_conditional_edges(
                    source,
                    router,
                    path_map,  # type: ignore[arg-type]
                )
            elif edge_def.target is not None:
                target = edge_def.target
                if target == "__end__":
                    target = END
                graph.add_edge(source, target)

        return graph.compile(checkpointer=checkpointer)

    async def _build_node(
        self,
        node_def: NodeDef,
        cel: CelEngine,
        config: DeclarativeConfig,
        *,
        has_skills: bool = False,
    ) -> Any:
        """Build a node function from a node definition."""
        assert self._registries is not None

        ctx = NodeContext(
            config=config,
            node_def=node_def,
            hook_manager=self.hook_manager if self.hook_manager._executors else None,
            extra={"sub_agent_tool_ids": self._sub_agent_tool_ids},
            registries=self._registries,
        )

        if node_def.type == "call_llm":
            args: CallLLMArgs = node_def.args  # type: ignore[assignment]
            llm_ref = args.llm or config.default_llm
            if llm_ref is None:
                raise GraphConstructionError(
                    f"Node '{node_def.name}' has no 'llm' and no "
                    f"'default_llm' is configured."
                )
            llm_id = llm_ref.id
            if llm_id not in self._registries.chat_models:
                raise GraphConstructionError(
                    f"Chat model '{llm_id}' not found. "
                    f"Ensure llm '{llm_id}' is declared in the YAML "
                    f"with a valid provider."
                )
            chat_model = self._registries.chat_models[llm_id]
            return build_call_llm_node(
                ctx,
                chat_model,
                cel,
                tool_registry=self._registries.tool_registry,
            )

        if node_def.type == "tool_node":
            return build_tool_node(
                ctx,
                tool_registry=self._registries.tool_registry,
                skill_registry=(
                    self._registries.skill_registry if has_skills else None
                ),
            )

        if node_def.type == "call_agent":
            ca_args: CallAgentArgs = node_def.args  # type: ignore[assignment]
            agent = await self._registries.agent_registry.get(
                ca_args.agent.id, ca_args.agent.version
            )
            return build_call_agent_node(ctx, agent, cel)

        if node_def.type == "data_transform":
            return build_data_transform_node(ctx, cel)

        if node_def.type == "set_state":
            return build_set_state_node(ctx, cel)

        if node_def.type == "interrupt":
            return build_interrupt_node(ctx, cel)

        if node_def.type == "load_skills":
            return build_load_skills_node(
                ctx,
                cel,
                skill_registry=self._registries.skill_registry,
                tool_registry=self._registries.tool_registry,
            )

        if node_def.type == "custom":
            return build_custom_node(ctx)

        raise GraphConstructionError(f"Unknown node type: {node_def.type}")

    def _build_cel_extra_vars(self, config: DeclarativeConfig) -> dict[str, Any]:
        """Build extra variables for CEL from config registries."""
        extra: dict[str, Any] = {}

        prompts: dict[str, dict[str, str]] = {}
        for p in config.prompts:
            prompts[p.id] = {"instructions": p.instructions or ""}
        if prompts:
            extra["prompts"] = prompts

        llms: dict[str, dict[str, str]] = {}
        for llm in config.llms:
            llms[llm.id] = {"model_name": llm.model_name}
        if llms:
            extra["llms"] = llms

        # Expose skill metadata so prompts can reference it without an
        # LLM tool call (e.g. ``skills["weather"]["name"]``).
        if config.skills and self._registries is not None:
            skills: dict[str, dict[str, str]] = {}
            skill_registry = self._registries.skill_registry
            for skill_def in config.skills:
                entries = skill_registry._entries.get(skill_def.id, {})
                entry = entries.get(skill_def.version)
                if entry and entry.instance:
                    fm = entry.instance.front_matter
                    skills[skill_def.id] = {
                        "id": skill_def.id,
                        "version": skill_def.version,
                        "name": fm.name,
                        "description": fm.description,
                    }
                else:
                    skills[skill_def.id] = {
                        "id": skill_def.id,
                        "version": skill_def.version,
                    }
            extra["skills"] = skills

        return extra
