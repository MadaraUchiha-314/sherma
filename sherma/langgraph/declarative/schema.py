"""Pydantic models for declarative agent YAML schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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


class CallLLMArgs(BaseModel):
    """Arguments for a call_llm node."""

    llm: RegistryRef
    prompt: str
    tools: list[RegistryRef] | None = None
    use_tools_from_registry: bool = False
    use_tools_from_loaded_skills: bool = False
    use_sub_agents_as_tools: bool = False
    response_format: ResponseFormatDef | None = None


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

    The interrupt value is the last ``AIMessage`` from state when
    available.  Falls back to the ``value`` CEL expression when no
    AIMessage is present.
    """

    value: str | None = None


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
    ]
    args: (
        CallLLMArgs
        | ToolNodeArgs
        | CallAgentArgs
        | DataTransformArgs
        | SetStateArgs
        | InterruptArgs
    )


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
    """Graph definition with nodes, edges, and entry point."""

    entry_point: str
    nodes: list[NodeDef]
    edges: list[EdgeDef]


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
    """A prompt declaration in the YAML."""

    id: str
    version: str = "*"
    instructions: str


class SkillDef(BaseModel):
    """A skill declaration in the YAML."""

    id: str
    version: str = "*"
    url: str | None = None
    skill_card_path: str | None = None


class HookDef(BaseModel):
    """A hook executor declaration in the YAML."""

    import_path: str


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


class CheckpointerDef(BaseModel):
    """Checkpointer configuration."""

    type: Literal["memory"] = "memory"


class DeclarativeConfig(BaseModel):
    """Top-level declarative configuration parsed from YAML."""

    agents: dict[str, AgentDef] = Field(default_factory=dict)
    llms: list[LLMDef] = Field(default_factory=list)
    tools: list[ToolDef] = Field(default_factory=list)
    prompts: list[PromptDef] = Field(default_factory=list)
    skills: list[SkillDef] = Field(default_factory=list)
    hooks: list[HookDef] = Field(default_factory=list)
    sub_agents: list[SubAgentDef] = Field(default_factory=list)
    checkpointer: CheckpointerDef | None = None
