"""Hook types: enum and context dataclasses for lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sherma.langgraph.declarative.nodes import NodeContext


class HookType(Enum):
    """Identifies each lifecycle hook point."""

    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_AGENT_CALL = "before_agent_call"
    AFTER_AGENT_CALL = "after_agent_call"
    BEFORE_SKILL_LOAD = "before_skill_load"
    AFTER_SKILL_LOAD = "after_skill_load"
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    BEFORE_INTERRUPT = "before_interrupt"
    AFTER_INTERRUPT = "after_interrupt"
    ON_CHAT_MODEL_CREATE = "on_chat_model_create"
    ON_GRAPH_INVOKE = "on_graph_invoke"


@dataclass
class BeforeLLMCallContext:
    """Context for before_llm_call hooks."""

    node_context: NodeContext
    node_name: str
    messages: list[Any]
    system_prompt: str
    tools: list[Any]
    state: dict[str, Any]


@dataclass
class AfterLLMCallContext:
    """Context for after_llm_call hooks."""

    node_context: NodeContext
    node_name: str
    response: Any
    state: dict[str, Any]


@dataclass
class BeforeToolCallContext:
    """Context for before_tool_call hooks."""

    node_context: NodeContext
    node_name: str
    tool_calls: list[Any]
    tools: list[Any]
    state: dict[str, Any]


@dataclass
class AfterToolCallContext:
    """Context for after_tool_call hooks."""

    node_context: NodeContext
    node_name: str
    result: dict[str, Any]
    state: dict[str, Any]


@dataclass
class BeforeAgentCallContext:
    """Context for before_agent_call hooks."""

    node_context: NodeContext
    node_name: str
    input_value: Any
    agent: Any
    state: dict[str, Any]


@dataclass
class AfterAgentCallContext:
    """Context for after_agent_call hooks."""

    node_context: NodeContext
    node_name: str
    result: dict[str, Any]
    state: dict[str, Any]


@dataclass
class BeforeSkillLoadContext:
    """Context for before_skill_load hooks."""

    node_context: NodeContext | None
    skill_id: str
    version: str


@dataclass
class AfterSkillLoadContext:
    """Context for after_skill_load hooks."""

    node_context: NodeContext | None
    skill_id: str
    version: str
    content: str
    tools_loaded: list[str] = field(default_factory=list)


@dataclass
class NodeEnterContext:
    """Context for node_enter hooks."""

    node_context: NodeContext
    node_name: str
    node_type: str
    state: dict[str, Any]


@dataclass
class NodeExitContext:
    """Context for node_exit hooks."""

    node_context: NodeContext
    node_name: str
    node_type: str
    result: dict[str, Any]
    state: dict[str, Any]


@dataclass
class BeforeInterruptContext:
    """Context for before_interrupt hooks."""

    node_context: NodeContext
    node_name: str
    value: Any
    state: dict[str, Any]


@dataclass
class AfterInterruptContext:
    """Context for after_interrupt hooks."""

    node_context: NodeContext
    node_name: str
    value: Any
    response: Any
    state: dict[str, Any]


@dataclass
class ChatModelCreateContext:
    """Context for on_chat_model_create hooks."""

    llm_id: str
    provider: str
    model_name: str
    kwargs: dict[str, Any]
    chat_model: Any | None = None


@dataclass
class GraphInvokeContext:
    """Context for on_graph_invoke hooks."""

    agent_id: str
    thread_id: str
    config: dict[str, Any]
    input: dict[str, Any]
