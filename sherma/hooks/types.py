"""Hook types: enum and context dataclasses for lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sherma.langgraph.declarative.nodes import NodeContext
    from sherma.langgraph.declarative.schema import CheckpointerDef
    from sherma.registry.bundle import RegistryBundle


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
    BEFORE_SKILL_UNLOAD = "before_skill_unload"
    AFTER_SKILL_UNLOAD = "after_skill_unload"
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    BEFORE_INTERRUPT = "before_interrupt"
    AFTER_INTERRUPT = "after_interrupt"
    ON_CHAT_MODEL_CREATE = "on_chat_model_create"
    ON_CHECKPOINTER_CREATE = "on_checkpointer_create"
    BEFORE_GRAPH_INVOKE = "before_graph_invoke"
    AFTER_GRAPH_INVOKE = "after_graph_invoke"
    NODE_EXECUTE = "node_execute"
    ON_NODE_ERROR = "on_node_error"
    ON_ERROR = "on_error"


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
class BeforeSkillUnloadContext:
    """Context for before_skill_unload hooks."""

    node_context: NodeContext | None
    skill_id: str
    version: str


@dataclass
class AfterSkillUnloadContext:
    """Context for after_skill_unload hooks."""

    node_context: NodeContext | None
    skill_id: str
    version: str
    tools_unloaded: list[str] = field(default_factory=list)


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
    """Context for on_chat_model_create hooks.

    ``chat_model`` accepts either a ready-to-use chat model instance
    **or** a zero-arg callable (factory) that returns one.  When a
    callable is provided, the model is constructed lazily on first use
    so that expensive setup (secrets, network) is deferred.
    """

    llm_id: str
    provider: str
    model_name: str
    kwargs: dict[str, Any]
    chat_model: Any | None = None  # BaseChatModel | Callable[[], BaseChatModel] | None


@dataclass
class CheckpointerCreateContext:
    """Context for on_checkpointer_create hooks.

    Runs once, while ``DeclarativeAgent`` is resolving its checkpointer
    from YAML.  Hooks can:

    * Mutate ``definition`` to rewrite the checkpointer config before
      the default builder runs (e.g. rewrite the ``url`` field to inject
      credentials fetched from Vault or AWS Secrets Manager).
    * Return a ready-to-use ``BaseCheckpointSaver`` instance by setting
      ``checkpointer`` — this short-circuits the default
      ``from_conn_string`` path entirely.

    ``definition`` is ``None`` when the YAML has no ``checkpointer:``
    block; hooks may still return a custom ``checkpointer`` in that
    case to install one programmatically.
    """

    definition: CheckpointerDef | None
    checkpointer: Any | None = None  # BaseCheckpointSaver | None


@dataclass
class GraphInvokeContext:
    """Context for before_graph_invoke hooks."""

    agent_id: str
    thread_id: str
    config: dict[str, Any]
    input: dict[str, Any]


@dataclass
class AfterGraphInvokeContext:
    """Context for after_graph_invoke hooks."""

    agent_id: str
    thread_id: str
    config: dict[str, Any]
    input: dict[str, Any]
    result: dict[str, Any]


@dataclass
class NodeExecuteContext:
    """Context for node_execute hooks (custom nodes only).

    ``result`` starts as an empty dict.  The hook populates it with
    the state updates the custom node should produce.

    ``registries`` is the per-tenant :class:`RegistryBundle` built for
    the agent, giving custom-node hooks direct access to chat models,
    tools, prompts, skills, and sub-agents at execution time (e.g. to
    invoke an LLM for summarisation or resolve a tool dynamically).
    It is ``None`` only in isolated unit tests that construct the
    context directly.  Remote (JSON-RPC) hooks never receive this
    field because it contains live Python objects.
    """

    node_context: NodeContext
    node_name: str
    state: dict[str, Any]
    result: dict[str, Any] = field(default_factory=dict)
    registries: RegistryBundle | None = None


@dataclass
class OnNodeErrorContext:
    """Context for on_node_error hooks."""

    node_context: NodeContext
    node_name: str
    node_type: str
    error: BaseException | None
    state: dict[str, Any]


@dataclass
class OnErrorContext:
    """Context for on_error hooks."""

    agent_id: str
    thread_id: str
    config: dict[str, Any]
    input: dict[str, Any]
    error: BaseException | None
