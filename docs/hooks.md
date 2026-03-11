# Hooks

Hooks give you programmatic control over the agent lifecycle. They let you observe, modify, or intercept behavior at every stage -- LLM calls, tool execution, agent invocation, skill loading, node transitions, and interrupts.

## Hook Types

sherma provides 14 lifecycle hook points:

| Hook | When it fires |
| --- | --- |
| `before_llm_call` | Before sending messages to the LLM |
| `after_llm_call` | After receiving the LLM response |
| `before_tool_call` | Before executing tool calls |
| `after_tool_call` | After tool execution completes |
| `before_agent_call` | Before invoking a sub-agent |
| `after_agent_call` | After receiving sub-agent response |
| `before_skill_load` | Before loading a skill via `load_skill_md` |
| `after_skill_load` | After a skill is loaded and its tools registered |
| `node_enter` | When execution enters any graph node |
| `node_exit` | When execution leaves any graph node |
| `before_interrupt` | Before an interrupt pauses graph execution |
| `after_interrupt` | After an interrupt resumes with user input |
| `on_chat_model_create` | When a chat model is being instantiated |
| `on_graph_invoke` | Before the LangGraph state graph is invoked |

## HookExecutor Protocol

A hook executor implements one or more hook methods. sherma defines a `HookExecutor` protocol and a `BaseHookExecutor` base class:

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import BeforeLLMCallContext, AfterLLMCallContext

class MyHook(BaseHookExecutor):
    # Override only the hooks you need.
    # All others return None (no-op) by default.

    async def before_llm_call(self, ctx: BeforeLLMCallContext) -> BeforeLLMCallContext | None:
        # Inspect or modify the context
        print(f"LLM call with {len(ctx.messages)} messages")
        return None  # Pass through unchanged

    async def after_llm_call(self, ctx: AfterLLMCallContext) -> AfterLLMCallContext | None:
        print(f"LLM responded: {ctx.response.content[:100]}")
        return None
```

### Return Values

- Return `None` to pass the context through unchanged
- Return a modified context to replace it for subsequent hooks

Multiple hook executors run in registration order. Each executor receives the context (possibly modified by previous executors).

## Context Objects

Each hook receives a typed context dataclass with relevant data:

### `BeforeLLMCallContext`

```python
@dataclass
class BeforeLLMCallContext:
    node_context: NodeContext   # Node metadata and config
    node_name: str              # Name of the current node
    messages: list[Any]         # Messages being sent to the LLM
    system_prompt: str          # System prompt text
    tools: list[Any]            # Tools bound to the LLM
    state: dict[str, Any]       # Current graph state
```

### `AfterLLMCallContext`

```python
@dataclass
class AfterLLMCallContext:
    node_context: NodeContext
    node_name: str
    response: Any              # LLM response (AIMessage)
    state: dict[str, Any]
```

### `BeforeToolCallContext`

```python
@dataclass
class BeforeToolCallContext:
    node_context: NodeContext
    node_name: str
    tool_calls: list[Any]      # Pending tool calls from the AIMessage
    tools: list[Any]           # Resolved tool implementations
    state: dict[str, Any]
```

### `AfterToolCallContext`

```python
@dataclass
class AfterToolCallContext:
    node_context: NodeContext
    node_name: str
    result: dict[str, Any]     # Tool execution result
    state: dict[str, Any]
```

### `NodeEnterContext` / `NodeExitContext`

```python
@dataclass
class NodeEnterContext:
    node_context: NodeContext
    node_name: str
    node_type: str             # "call_llm", "tool_node", etc.
    state: dict[str, Any]

@dataclass
class NodeExitContext:
    node_context: NodeContext
    node_name: str
    node_type: str
    result: dict[str, Any]     # Node output
    state: dict[str, Any]
```

### `BeforeInterruptContext` / `AfterInterruptContext`

```python
@dataclass
class BeforeInterruptContext:
    node_context: NodeContext
    node_name: str
    value: Any                 # Interrupt value (message to user)
    state: dict[str, Any]

@dataclass
class AfterInterruptContext:
    node_context: NodeContext
    node_name: str
    value: Any
    response: Any              # User's response
    state: dict[str, Any]
```

### Skill Load Contexts

```python
@dataclass
class BeforeSkillLoadContext:
    node_context: NodeContext | None
    skill_id: str
    version: str

@dataclass
class AfterSkillLoadContext:
    node_context: NodeContext | None
    skill_id: str
    version: str
    content: str               # SKILL.md content
    tools_loaded: list[str]    # IDs of tools registered
```

### `ChatModelCreateContext`

```python
@dataclass
class ChatModelCreateContext:
    llm_id: str                # Registry ID of the LLM
    provider: str              # Provider name (e.g. "openai")
    model_name: str            # Model name (e.g. "gpt-4o-mini")
    kwargs: dict[str, Any]     # Constructor kwargs passed to the chat model
    chat_model: Any | None = None  # Set to override the chat model instance
```

The `on_chat_model_create` hook fires when a chat model is being instantiated from an LLM definition. Use it to customize model creation -- swap the model, inject API keys, or provide a fully constructed chat model instance.

**Example: inject an API key and swap model**

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import ChatModelCreateContext

class ChatModelConfigHook(BaseHookExecutor):
    async def on_chat_model_create(
        self, ctx: ChatModelCreateContext
    ) -> ChatModelCreateContext | None:
        # Inject a custom API key
        ctx.kwargs["api_key"] = get_secret("OPENAI_API_KEY")

        # Swap to a different model at runtime
        ctx.model_name = "gpt-4o"

        return ctx  # Return modified context
```

**Example: provide a pre-built chat model**

```python
from langchain_openai import ChatOpenAI
from sherma import BaseHookExecutor
from sherma.hooks.types import ChatModelCreateContext

class CustomChatModelHook(BaseHookExecutor):
    async def on_chat_model_create(
        self, ctx: ChatModelCreateContext
    ) -> ChatModelCreateContext | None:
        # Supply a fully constructed chat model -- skips default creation
        ctx.chat_model = ChatOpenAI(model="gpt-4o", temperature=0)
        return ctx
```

### `GraphInvokeContext`

```python
@dataclass
class GraphInvokeContext:
    agent_id: str              # ID of the agent being invoked
    thread_id: str             # Thread ID for the conversation
    config: dict[str, Any]     # The full RunnableConfig dict (mutable)
    input: dict[str, Any]      # The input being passed to ainvoke
```

The `on_graph_invoke` hook fires just before `graph.ainvoke()` is called in `LangGraphAgent.send_message()`. Use it to customize the LangGraph `RunnableConfig` -- change the recursion limit, add custom configurable keys, set callbacks, etc.

**Example: increase recursion limit and add custom configurable**

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import GraphInvokeContext

class GraphConfigHook(BaseHookExecutor):
    async def on_graph_invoke(
        self, ctx: GraphInvokeContext
    ) -> GraphInvokeContext | None:
        ctx.config["recursion_limit"] = 50
        ctx.config["configurable"]["user_id"] = "user-123"
        return ctx
```

**Example: add LangChain callbacks**

```python
from langchain_core.callbacks import StdOutCallbackHandler
from sherma import BaseHookExecutor
from sherma.hooks.types import GraphInvokeContext

class CallbackHook(BaseHookExecutor):
    async def on_graph_invoke(
        self, ctx: GraphInvokeContext
    ) -> GraphInvokeContext | None:
        ctx.config["callbacks"] = [StdOutCallbackHandler()]
        return ctx
```

## Registering Hooks

### Programmatic (on any LangGraphAgent)

```python
from sherma.langgraph.agent import LangGraphAgent

agent = MyAgent(id="my-agent", version="1.0.0")
agent.register_hooks(LoggingHook())
agent.register_hooks(GuardrailHook())
```

### Declarative (via constructor)

```python
from sherma import DeclarativeAgent

agent = DeclarativeAgent(
    id="my-agent",
    version="1.0.0",
    yaml_path="agent.yaml",
    hooks=[LoggingHook(), GuardrailHook()],
)
```

### Declarative (via YAML)

```yaml
hooks:
  - import_path: my_package.hooks.LoggingHook
  - import_path: my_package.hooks.GuardrailHook
```

Hooks from the constructor and YAML are both registered. Constructor hooks run first.

## Example: Logging Hook

```python
import time
from sherma import BaseHookExecutor
from sherma.hooks.types import (
    BeforeLLMCallContext, AfterLLMCallContext,
    BeforeToolCallContext, AfterToolCallContext,
    NodeEnterContext, NodeExitContext,
)

class LoggingHook(BaseHookExecutor):
    async def node_enter(self, ctx: NodeEnterContext) -> NodeEnterContext | None:
        print(f">>> Entering '{ctx.node_name}' (type={ctx.node_type})")
        ctx.state["__enter_time__"] = time.monotonic()
        return None

    async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
        enter_time = ctx.state.get("__enter_time__")
        elapsed = f" ({time.monotonic() - enter_time:.3f}s)" if enter_time else ""
        print(f"<<< Exiting '{ctx.node_name}'{elapsed}")
        return None

    async def before_llm_call(self, ctx: BeforeLLMCallContext) -> BeforeLLMCallContext | None:
        print(f"LLM: {len(ctx.messages)} msgs, {len(ctx.tools)} tools")
        return None

    async def before_tool_call(self, ctx: BeforeToolCallContext) -> BeforeToolCallContext | None:
        names = [tc.get("name", "?") for tc in ctx.tool_calls]
        print(f"Tools: {names}")
        return None
```

## Example: Prompt Guardrail Hook

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import BeforeLLMCallContext

class PromptGuardrailHook(BaseHookExecutor):
    GUARDRAIL = (
        "\n\nIMPORTANT: Always be helpful, accurate, and concise. "
        "Never fabricate data -- if a tool returns an error, say so."
    )

    async def before_llm_call(self, ctx: BeforeLLMCallContext) -> BeforeLLMCallContext | None:
        ctx.system_prompt += self.GUARDRAIL
        return ctx  # Return modified context
```

## HookManager

The `HookManager` orchestrates hook execution. It's managed automatically by `LangGraphAgent`, but you can also use it directly:

```python
from sherma import HookManager

manager = HookManager()
manager.register(LoggingHook())
manager.register(GuardrailHook())

# Run a hook through all registered executors
ctx = await manager.run_hook("before_llm_call", before_ctx)
```

Hooks execute in registration order. If an executor returns a modified context, that becomes the input for the next executor.
