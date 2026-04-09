"""Pydantic models for declarative agent YAML schema."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from sherma.langgraph.declarative.env import expand_env_vars


class RegistryRef(BaseModel):
    """Reference to a registered entity."""

    id: str
    version: str = "*"


class StateFieldDef(BaseModel):
    """A single field in the agent state schema."""

    name: str
    type: str = "str"
    default: Any = None


class StateDef(BaseModel):
    """State schema definition."""

    fields: list[StateFieldDef] = Field(default_factory=list)


class ResponseFormatDef(BaseModel):
    """Structured output schema for call_llm."""

    name: str
    description: str = ""
    schema_: dict[str, Any] = Field(alias="schema")


class PromptMessageDef(BaseModel):
    """A single message in an array prompt."""

    role: Literal["system", "human", "ai", "messages"]
    content: str  # CEL expression


class CallLLMArgs(BaseModel):
    """Arguments for a call_llm node."""

    llm: RegistryRef | None = None
    prompt: list[PromptMessageDef]
    tools: list[RegistryRef] | None = None
    use_tools_from_registry: bool = False
    use_tools_from_loaded_skills: bool = False
    use_sub_agents_as_tools: Literal[False, "all"] | list[RegistryRef] = False
    response_format: ResponseFormatDef | None = None
    state_updates: dict[str, str]

    @field_validator("use_sub_agents_as_tools", mode="before")
    @classmethod
    def _normalize_use_sub_agents_as_tools(
        cls, v: Any
    ) -> Literal[False, "all"] | list[RegistryRef]:
        if v is True:
            return "all"
        if v is False:
            return False
        if v == "all":
            return "all"
        if isinstance(v, list):
            return v  # Pydantic validates as list[RegistryRef]
        raise ValueError(
            f"use_sub_agents_as_tools must be true, false, 'all', or a list "
            f"of RegistryRef objects, got {v!r}"
        )


class ToolNodeArgs(BaseModel):
    """Arguments for a tool_node node.

    The tool_node executes whatever tool calls the LLM placed in the last
    AIMessage.  It resolves tool implementations from the ToolRegistry at
    invocation time — no explicit tool list is needed.

    An explicit ``tools`` list may still be provided to restrict execution
    to a specific subset of registry tools.
    """

    tools: list[RegistryRef] | None = None


class CallAgentArgs(BaseModel):
    """Arguments for a call_agent node."""

    agent: RegistryRef
    input: str


class DataTransformArgs(BaseModel):
    """Arguments for a data_transform node."""

    expression: str


class SetStateArgs(BaseModel):
    """Arguments for a set_state node."""

    values: dict[str, str]


class InterruptArgs(BaseModel):
    """Arguments for an interrupt node.

    The interrupt value is always the result of evaluating the ``value``
    CEL expression against the current state.
    """

    value: str


class LoadSkillsArgs(BaseModel):
    """Arguments for a load_skills node.

    The ``skill_ids`` CEL expression must evaluate to a list of objects
    with ``id`` (required) and ``version`` (optional, defaults to ``"*"``)
    keys.
    """

    skill_ids: str


class CustomArgs(BaseModel):
    """Arguments for a custom node.

    Custom nodes have no built-in logic — their behaviour is defined
    entirely by the ``node_execute`` hook.  An optional ``metadata``
    dict can carry arbitrary data accessible to hooks via
    ``ctx.node_context.node_def.args.metadata``.
    """

    metadata: dict[str, Any] = Field(default_factory=dict)


class RetryPolicy(BaseModel):
    """Retry configuration for a node.

    Controls how many times the retryable operation is attempted and
    the delay between attempts.  ``max_attempts`` is the **total**
    number of tries (1 initial + retries).
    """

    max_attempts: int = 3
    strategy: Literal["fixed", "exponential"] = "exponential"
    delay: float = 1.0  # base delay in seconds
    max_delay: float = 30.0  # cap for exponential backoff


class OnErrorDef(BaseModel):
    """Declarative error handling for a node.

    * ``retry`` - retry policy (only supported on ``call_llm`` nodes).
    * ``fallback`` - name of a node to route to when the error is not
      recovered by retries.  Supported on ``call_llm``, ``tool_node``,
      and ``call_agent`` nodes.
    """

    retry: RetryPolicy | None = None
    fallback: str | None = None


class NodeDef(BaseModel):
    """A node definition in the graph."""

    name: str
    type: Literal[
        "call_llm",
        "tool_node",
        "call_agent",
        "data_transform",
        "set_state",
        "interrupt",
        "load_skills",
        "custom",
    ]
    args: (
        CallLLMArgs
        | ToolNodeArgs
        | CallAgentArgs
        | DataTransformArgs
        | SetStateArgs
        | InterruptArgs
        | LoadSkillsArgs
        | CustomArgs
    )
    on_error: OnErrorDef | None = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_args_type(cls, data: Any) -> Any:
        """Parse args using the correct model based on node type."""
        if not isinstance(data, dict):
            return data
        node_type = data.get("type")
        raw_args = data.get("args")
        if node_type is None or raw_args is None or not isinstance(raw_args, dict):
            return data
        type_map: dict[str, type[BaseModel]] = {
            "call_llm": CallLLMArgs,
            "tool_node": ToolNodeArgs,
            "call_agent": CallAgentArgs,
            "data_transform": DataTransformArgs,
            "set_state": SetStateArgs,
            "interrupt": InterruptArgs,
            "load_skills": LoadSkillsArgs,
            "custom": CustomArgs,
        }
        args_cls = type_map.get(node_type)
        if args_cls is not None:
            data = {**data, "args": args_cls(**raw_args)}
        return data


class BranchDef(BaseModel):
    """A conditional branch in an edge."""

    condition: str
    target: str


class EdgeDef(BaseModel):
    """An edge definition in the graph."""

    source: str
    target: str | None = None
    branches: list[BranchDef] | None = None
    default: str | None = None


class GraphDef(BaseModel):
    """Graph definition with nodes, edges, and entry point.

    Exactly one of the following must be provided:

    * ``entry_point`` — name of the node that `__start__` unconditionally
      routes to.
    * one or more edges in ``edges`` whose ``source`` is ``"__start__"``.
      Such edges may be either static or conditional, enabling branching
      directly at graph entry.
    """

    entry_point: str | None = None
    nodes: list[NodeDef]
    edges: list[EdgeDef]

    @model_validator(mode="after")
    def _check_entry(self) -> Self:
        has_start_edge = any(e.source == "__start__" for e in self.edges)
        if self.entry_point is None and not has_start_edge:
            raise ValueError(
                "GraphDef requires either 'entry_point' or at least one "
                "edge with source '__start__'"
            )
        if self.entry_point is not None and has_start_edge:
            raise ValueError(
                "GraphDef cannot have both 'entry_point' and edges with "
                "source '__start__'"
            )
        return self


class LangGraphConfigDef(BaseModel):
    """LangGraph runtime configuration settable per-agent in YAML."""

    recursion_limit: int | None = None
    max_concurrency: int | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class AgentDef(BaseModel):
    """A single agent definition."""

    state: StateDef
    graph: GraphDef
    langgraph_config: LangGraphConfigDef | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


class LLMDef(BaseModel):
    """An LLM declaration in the YAML."""

    id: str
    version: str = "*"
    provider: str = "openai"
    model_name: str


class ToolDef(BaseModel):
    """A tool declaration in the YAML.

    Use import_path to reference a LangGraph/LangChain tool directly
    from Python, e.g. "examples.tools.get_weather".
    """

    id: str
    version: str = "*"
    import_path: str | None = None
    url: str | None = None
    protocol: str | None = None


class PromptDef(BaseModel):
    """A prompt declaration in the YAML.

    Provide exactly one of ``instructions`` (inline string) or
    ``instructions_path`` (path to a file whose contents become the
    prompt instructions). Relative ``instructions_path`` values are
    resolved against the YAML's ``base_path``.
    """

    id: str
    version: str = "*"
    instructions: str | None = None
    instructions_path: str | None = None

    @model_validator(mode="after")
    def _check_one_source(self) -> Self:
        if self.instructions is None and self.instructions_path is None:
            raise ValueError(
                "PromptDef requires either 'instructions' or 'instructions_path'"
            )
        if self.instructions is not None and self.instructions_path is not None:
            raise ValueError(
                "PromptDef cannot have both 'instructions' and 'instructions_path'"
            )
        return self


class SkillDef(BaseModel):
    """A skill declaration in the YAML."""

    id: str
    version: str = "*"
    url: str | None = None
    skill_card_path: str | None = None


class HookDef(BaseModel):
    """A hook executor declaration in the YAML.

    Provide either ``import_path`` for a local Python hook executor
    or ``url`` for a remote JSON-RPC 2.0 hook server.
    """

    import_path: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _check_one_source(self) -> Self:
        if not self.import_path and not self.url:
            raise ValueError("HookDef requires either 'import_path' or 'url'")
        if self.import_path and self.url:
            raise ValueError("HookDef cannot have both 'import_path' and 'url'")
        return self


class SubAgentDef(BaseModel):
    """A sub-agent declaration in the YAML.

    Sub-agents can be remote (via ``url``), local Python objects
    (via ``import_path``), or declarative YAML agents (via ``yaml_path``).
    If none of these are provided, the agent is expected to already be
    registered in the agent registry.
    """

    id: str
    version: str = "*"
    url: str | None = None
    import_path: str | None = None
    yaml_path: str | None = None


class MemoryCheckpointerDef(BaseModel):
    """In-memory checkpointer configuration.

    Wraps LangGraph's ``MemorySaver``.  Stateless — nothing persists
    across process restarts.
    """

    type: Literal["memory"] = "memory"


class RedisCheckpointerDef(BaseModel):
    """Redis-backed checkpointer configuration.

    Wraps ``langgraph-checkpoint-redis``'s ``AsyncRedisSaver``.  The
    optional ``sherma[redis]`` extra must be installed.

    ``url`` is an ordinary Redis URL
    (``redis://[:password@]host:port[/db]``) and supports
    ``${VAR}`` / ``${VAR:-default}`` environment-variable
    interpolation.
    """

    type: Literal["redis"]
    url: str
    ttl_minutes: int | None = None

    @field_validator("url")
    @classmethod
    def _expand_url(cls, v: str) -> str:
        expanded = expand_env_vars(v)
        if not expanded:
            raise ValueError("Redis checkpointer 'url' must not be empty")
        return expanded


class PostgresCheckpointerDef(BaseModel):
    """PostgreSQL-backed checkpointer configuration.

    Wraps ``langgraph-checkpoint-postgres``'s ``AsyncPostgresSaver``.
    The optional ``sherma[postgres]`` extra must be installed.

    ``url`` is an ordinary Postgres URL
    (``postgresql://user:password@host:port/db``) and supports
    ``${VAR}`` / ``${VAR:-default}`` environment-variable
    interpolation.
    """

    type: Literal["postgres"]
    url: str

    @field_validator("url")
    @classmethod
    def _expand_url(cls, v: str) -> str:
        expanded = expand_env_vars(v)
        if not expanded:
            raise ValueError("Postgres checkpointer 'url' must not be empty")
        return expanded


CheckpointerDef = Annotated[
    MemoryCheckpointerDef | RedisCheckpointerDef | PostgresCheckpointerDef,
    Field(discriminator="type"),
]


class DeclarativeConfig(BaseModel):
    """Top-level declarative configuration parsed from YAML."""

    manifest_version: int
    agents: dict[str, AgentDef] = Field(default_factory=dict)
    llms: list[LLMDef] = Field(default_factory=list)
    tools: list[ToolDef] = Field(default_factory=list)
    prompts: list[PromptDef] = Field(default_factory=list)
    skills: list[SkillDef] = Field(default_factory=list)
    hooks: list[HookDef] = Field(default_factory=list)
    sub_agents: list[SubAgentDef] = Field(default_factory=list)
    default_llm: RegistryRef | None = None
    checkpointer: CheckpointerDef | None = None
