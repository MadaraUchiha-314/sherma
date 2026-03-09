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


class CallLLMArgs(BaseModel):
    """Arguments for a call_llm node."""

    llm: RegistryRef
    prompt: str
    tools: list[RegistryRef] | None = None


class ToolNodeArgs(BaseModel):
    """Arguments for a tool_node node.

    Tools are optional. When omitted, the tool_node automatically inherits
    tools from call_llm nodes in the graph (since it executes whatever
    tool calls the LLM placed in the AIMessage).
    """

    tools: list[RegistryRef] | None = None
    tool_calls: str | None = None


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


class NodeDef(BaseModel):
    """A node definition in the graph."""

    name: str
    type: Literal["call_llm", "tool_node", "call_agent", "data_transform", "set_state"]
    args: CallLLMArgs | ToolNodeArgs | CallAgentArgs | DataTransformArgs | SetStateArgs


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


class AgentDef(BaseModel):
    """A single agent definition."""

    state: StateDef
    graph: GraphDef
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


class DeclarativeConfig(BaseModel):
    """Top-level declarative configuration parsed from YAML."""

    agents: dict[str, AgentDef] = Field(default_factory=dict)
    llms: list[LLMDef] = Field(default_factory=list)
    tools: list[ToolDef] = Field(default_factory=list)
    prompts: list[PromptDef] = Field(default_factory=list)
    skills: list[SkillDef] = Field(default_factory=list)
