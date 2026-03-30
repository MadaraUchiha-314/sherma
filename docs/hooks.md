# Hooks

Hooks give you programmatic control over the agent lifecycle. They let you observe, modify, or intercept behavior at every stage -- LLM calls, tool execution, agent invocation, skill loading, node transitions, and interrupts.

## Hook Types

sherma provides 20 lifecycle hook points:

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
| `before_skill_unload` | Before unloading a skill via `unload_skill_md` |
| `after_skill_unload` | After a skill is unloaded and its tools deregistered |
| `node_enter` | When execution enters any graph node |
| `node_execute` | When a `custom` node runs its logic (custom nodes only) |
| `node_exit` | When execution leaves any graph node |
| `before_interrupt` | Before an interrupt pauses graph execution |
| `after_interrupt` | After an interrupt resumes with user input |
| `on_chat_model_create` | When a chat model is being instantiated |
| `before_graph_invoke` | Before the LangGraph state graph is invoked |
| `after_graph_invoke` | After the LangGraph state graph completes |
| `on_node_error` | When a declarative node function raises an exception |
| `on_error` | When `graph.ainvoke()` raises an exception (catch-all) |

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

### `NodeEnterContext` / `NodeExecuteContext` / `NodeExitContext`

```python
@dataclass
class NodeEnterContext:
    node_context: NodeContext
    node_name: str
    node_type: str             # "call_llm", "tool_node", "custom", etc.
    state: dict[str, Any]

@dataclass
class NodeExecuteContext:
    """Fires only for custom nodes, between node_enter and node_exit."""
    node_context: NodeContext
    node_name: str
    state: dict[str, Any]
    result: dict[str, Any]     # Starts as {}, hook populates it

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

### Skill Unload Contexts

```python
@dataclass
class BeforeSkillUnloadContext:
    node_context: NodeContext | None
    skill_id: str
    version: str

@dataclass
class AfterSkillUnloadContext:
    node_context: NodeContext | None
    skill_id: str
    version: str
    tools_unloaded: list[str]  # IDs of tools deregistered
```

### `ChatModelCreateContext`

```python
@dataclass
class ChatModelCreateContext:
    llm_id: str                # Registry ID of the LLM
    provider: str              # Provider name (e.g. "openai")
    model_name: str            # Model name (e.g. "gpt-4o-mini")
    kwargs: dict[str, Any]     # Constructor kwargs passed to the chat model
    chat_model: Any | None = None  # Instance, or callable factory
```

The `on_chat_model_create` hook fires when a chat model is being instantiated from an LLM definition. Use it to customize model creation -- swap the model, inject API keys, or provide a fully constructed chat model instance.

#### `chat_model`: instance or factory

`chat_model` accepts either:

- **An instance** -- a ready-to-use chat model (e.g., `ChatOpenAI(...)`)
- **A callable factory** -- a zero-arg callable that returns a chat model

When a callable is provided, sherma wraps it in a `LazyChatModel` proxy that defers construction until the model is first used (i.e., on the first LLM call during request handling). This is important when model construction depends on infrastructure that isn't available at import time or during server startup -- for example, secret managers, auth token providers, or database connections that initialize asynchronously.

sherma distinguishes instances from factories by checking `callable(chat_model) and not hasattr(chat_model, "invoke")`. Any object with an `invoke` method is treated as a ready-to-use model.

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

**Example: lazy factory for deferred construction**

Use a factory when the chat model depends on infrastructure that initializes after import time (e.g., a secret manager, an auth service, or database-backed config):

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import ChatModelCreateContext

class LazyModelHook(BaseHookExecutor):
    async def on_chat_model_create(
        self, ctx: ChatModelCreateContext
    ) -> ChatModelCreateContext | None:
        if ctx.llm_id == "gpt-4o":
            model_name = ctx.model_name
            # Factory -- called lazily on first LLM invocation,
            # not during graph construction at startup.
            ctx.chat_model = lambda: create_authenticated_model(model_name)
        return ctx
```

The `LazyChatModel` proxy is transparent -- it forwards all attribute access and method calls to the real model once constructed. After the first access, the factory is never called again.

### `GraphInvokeContext`

```python
@dataclass
class GraphInvokeContext:
    agent_id: str              # ID of the agent being invoked
    thread_id: str             # Thread ID for the conversation
    config: dict[str, Any]     # The full RunnableConfig dict (mutable)
    input: dict[str, Any]      # The input being passed to ainvoke
```

The `before_graph_invoke` hook fires just before `graph.ainvoke()` is called in `LangGraphAgent.send_message()`. Use it to customize the LangGraph `RunnableConfig` -- change the recursion limit, add custom configurable keys, set callbacks, etc.

**Example: increase recursion limit and add custom configurable**

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import GraphInvokeContext

class GraphConfigHook(BaseHookExecutor):
    async def before_graph_invoke(
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
    async def before_graph_invoke(
        self, ctx: GraphInvokeContext
    ) -> GraphInvokeContext | None:
        ctx.config["callbacks"] = [StdOutCallbackHandler()]
        return ctx
```

### `AfterGraphInvokeContext`

```python
@dataclass
class AfterGraphInvokeContext:
    agent_id: str              # ID of the agent being invoked
    thread_id: str             # Thread ID for the conversation
    config: dict[str, Any]     # The RunnableConfig dict used for invocation
    input: dict[str, Any]      # The input that was passed to ainvoke
    result: dict[str, Any]     # The graph result (mutable)
```

The `after_graph_invoke` hook fires after `graph.ainvoke()` returns. Use it to inspect or modify the graph result before it is converted to an A2A response.

**Example: post-process the graph result**

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import AfterGraphInvokeContext

class PostProcessHook(BaseHookExecutor):
    async def after_graph_invoke(
        self, ctx: AfterGraphInvokeContext
    ) -> AfterGraphInvokeContext | None:
        # Log or modify the result
        print(f"Graph returned {len(ctx.result.get('messages', []))} messages")
        return ctx
```

### `OnNodeErrorContext`

```python
@dataclass
class OnNodeErrorContext:
    node_context: NodeContext
    node_name: str             # Name of the node that raised
    node_type: str             # "call_llm", "tool_node", etc.
    error: BaseException | None  # The exception (mutable)
    state: dict[str, Any]
```

The `on_node_error` hook fires when any declarative node function raises an exception. All six node types are covered: `call_llm`, `tool_node`, `call_agent`, `data_transform`, `set_state`, and `interrupt`.

#### Error hook semantics

The `error` field controls what happens after all hooks run:

- **Pass through** -- return `None` to leave the context unchanged (error continues to the next hook)
- **Consume** -- set `error = None` and return the context to swallow the error (node returns an empty dict as fallback)
- **Replace** -- set `error` to a different exception to replace the original

Multiple hooks chain in registration order. Each hook sees the `error` as left by the previous hook.

**Example: log and consume errors from a specific node**

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import OnNodeErrorContext

class NodeErrorHandler(BaseHookExecutor):
    async def on_node_error(
        self, ctx: OnNodeErrorContext
    ) -> OnNodeErrorContext | None:
        print(f"Node '{ctx.node_name}' ({ctx.node_type}) failed: {ctx.error}")

        # Swallow errors from the "summarize" node
        if ctx.node_name == "summarize":
            ctx.error = None
            return ctx

        # Let all other errors propagate
        return None
```

**Example: replace errors with a custom exception**

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import OnNodeErrorContext

class WrapNodeError(BaseHookExecutor):
    async def on_node_error(
        self, ctx: OnNodeErrorContext
    ) -> OnNodeErrorContext | None:
        ctx.error = RuntimeError(
            f"Node '{ctx.node_name}' failed: {ctx.error}"
        )
        return ctx
```

### `OnErrorContext`

```python
@dataclass
class OnErrorContext:
    agent_id: str              # ID of the agent
    thread_id: str             # Thread ID for the conversation
    config: dict[str, Any]     # The RunnableConfig dict
    input: dict[str, Any]      # The input that was passed to ainvoke
    error: BaseException | None  # The exception (mutable)
```

The `on_error` hook fires when `graph.ainvoke()` raises an exception in `LangGraphAgent.send_message()`. It acts as a catch-all for errors that escape individual nodes.

The `error` field follows the same semantics as `on_node_error`:

- **Pass through** -- return `None` (error continues)
- **Consume** -- set `error = None` (send_message returns without yielding any events)
- **Replace** -- set `error` to a different exception

**Example: catch-all error logging**

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import OnErrorContext

class GraphErrorLogger(BaseHookExecutor):
    async def on_error(
        self, ctx: OnErrorContext
    ) -> OnErrorContext | None:
        print(
            f"Agent '{ctx.agent_id}' graph invocation failed: {ctx.error}"
        )
        return None  # Let the error propagate
```

**Example: swallow errors and return gracefully**

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import OnErrorContext

class GracefulErrorHandler(BaseHookExecutor):
    async def on_error(
        self, ctx: OnErrorContext
    ) -> OnErrorContext | None:
        # Consume the error -- send_message returns without events.
        # The A2A executor will call task_updater.complete() with no message.
        ctx.error = None
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

### Declarative (via YAML, remote)

```yaml
hooks:
  - url: http://localhost:8000/hooks
```

You can mix local and remote hooks in the same config:

```yaml
hooks:
  - import_path: my_package.hooks.LoggingHook
  - url: http://localhost:8000/hooks
```

Hooks from the constructor and YAML are both registered. Constructor hooks run first, then YAML hooks, in declaration order. See [Remote Hooks (JSON-RPC)](#remote-hooks-json-rpc) for details.

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

## Remote Hooks (JSON-RPC)

Remote hooks let you implement hook logic in a separate service -- in any language or framework -- and connect it to sherma over HTTP using the [JSON-RPC 2.0](https://www.jsonrpc.org/specification) protocol.

This is useful when:

- You want to share hook logic across multiple agents or teams
- Your hook logic is in a different language (Node.js, Go, Java, etc.)
- You want to deploy and scale hooks independently from the agent

### How it works

When a remote hook is registered, sherma serializes each hook context to JSON and sends it as a JSON-RPC method call to your server. The method name is the hook name (e.g., `before_llm_call`). Your server processes the request and returns a (possibly modified) context.

```
Agent                        Hook Server
  |                              |
  |-- POST JSON-RPC ----------->|
  |   method: "before_llm_call" |
  |   params: { ... context }   |
  |                              |
  |<--- JSON-RPC result --------|
  |   result: { ... modified }  |
  |   (or result: null)         |
```

### Registering a remote hook

#### In YAML

```yaml
hooks:
  - url: http://localhost:8000/hooks
```

#### In Python

```python
from sherma import DeclarativeAgent, RemoteHookExecutor

agent = DeclarativeAgent(
    id="my-agent",
    version="1.0.0",
    yaml_path="agent.yaml",
    hooks=[RemoteHookExecutor(url="http://localhost:8000/hooks")],
)
```

You can mix local and remote hooks. They run in registration order just like local hooks.

```yaml
hooks:
  - import_path: my_package.hooks.LoggingHook
  - url: http://localhost:8000/hooks
```

### HookHandler

sherma provides a `HookHandler` base class -- the remote equivalent of `BaseHookExecutor`. Subclass it and override only the hooks you need. Each method receives a plain `dict` and returns a modified dict (or `None` to pass through):

```python
from sherma.hooks.handler import HookHandler

class MyHooks(HookHandler):
    async def before_llm_call(self, params):
        params["system_prompt"] += "\n\nBe concise and accurate."
        return params

    async def node_enter(self, params):
        print(f">>> Entering node '{params['node_name']}'")
        return None  # Observation only, no modification
```

`HookHandler` mirrors the `BaseHookExecutor` interface but works with JSON-serializable dicts instead of Python dataclasses. The `on_chat_model_create` hook is intentionally absent since it cannot work over JSON-RPC.

### HookFastAPIApplication / HookStarletteApplication

Pass your handler to an application builder to get a ready-to-run ASGI server. This is the same pattern as A2A's `A2AFastAPIApplication` / `A2AStarletteApplication`:

**FastAPI:**

```python
from sherma.hooks.apps import HookFastAPIApplication
from sherma.hooks.handler import HookHandler

class MyHooks(HookHandler):
    async def before_llm_call(self, params):
        params["system_prompt"] += "\n\nBe concise."
        return params

    async def before_graph_invoke(self, params):
        params["config"]["recursion_limit"] = 50
        return params

app = HookFastAPIApplication(handler=MyHooks()).build()
```

```bash
pip install fastapi uvicorn
uvicorn hook_server:app --port 8000
```

**Starlette:**

```python
from sherma.hooks.apps import HookStarletteApplication
from sherma.hooks.handler import HookHandler

class MyHooks(HookHandler):
    async def before_llm_call(self, params):
        params["system_prompt"] += "\n\nBe concise."
        return params

app = HookStarletteApplication(handler=MyHooks()).build()
```

```bash
pip install starlette uvicorn
uvicorn hook_server:app --port 8000
```

Then point your agent at it:

```yaml
hooks:
  - url: http://localhost:8000/hooks
```

### Adding hooks to an existing app

If you already have a FastAPI or Starlette application, use `add_routes_to_app` instead of `build`:

```python
from fastapi import FastAPI
from sherma.hooks.apps import HookFastAPIApplication

existing_app = FastAPI()
HookFastAPIApplication(handler=MyHooks()).add_routes_to_app(
    existing_app, rpc_url="/hooks"
)
```

```python
from starlette.applications import Starlette
from sherma.hooks.apps import HookStarletteApplication

existing_app = Starlette()
HookStarletteApplication(handler=MyHooks()).add_routes_to_app(
    existing_app, rpc_url="/hooks"
)
```

### JSON-RPC protocol details

Each hook call is a standard JSON-RPC 2.0 request:

```json
{
    "jsonrpc": "2.0",
    "method": "before_llm_call",
    "params": {
        "node_name": "agent",
        "system_prompt": "You are a helpful assistant.",
        "messages": [...],
        "tools": [...],
        "state": {"messages": [...]}
    },
    "id": 1
}
```

Your server should return one of:

**Modified context** -- sherma applies the changes:

```json
{
    "jsonrpc": "2.0",
    "result": {
        "node_name": "agent",
        "system_prompt": "You are a helpful assistant.\n\nBe concise.",
        "state": {"messages": [...]}
    },
    "id": 1
}
```

**Null result** -- pass through unchanged (equivalent to returning `None` in a local hook):

```json
{
    "jsonrpc": "2.0",
    "result": null,
    "id": 1
}
```

**Error** -- sherma logs a warning and passes through (the agent is never blocked by a failing hook server):

```json
{
    "jsonrpc": "2.0",
    "error": {"code": -32000, "message": "something went wrong"},
    "id": 1
}
```

### What the server receives and can modify

Not all fields from the hook context are sent over JSON-RPC. Some fields are Python-only objects that cannot be serialized.

| Field | Sent to server | Modifiable | Notes |
| --- | --- | --- | --- |
| `node_name`, `node_type`, `agent_id`, `thread_id`, `skill_id`, `version`, `content`, `tools_loaded` | Yes | Yes | Primitive fields |
| `system_prompt`, `input_value` | Yes | Yes | String fields |
| `state`, `config`, `input`, `result`, `kwargs` | Yes | Yes | Dict fields -- the main way to influence behavior |
| `messages`, `response`, `tools`, `tool_calls` | Yes (read-only) | No | Serialized for observation but kept from original on response |
| `node_context` | No | No | Internal framework object |
| `agent` | No | No | Python agent instance |
| `chat_model` | No | No | Python model instance |
| `error` | Yes (as `{"type": "...", "message": "..."}`) | No | Serialized for observation, original kept |

### Unsupported hooks

The `on_chat_model_create` hook is **not called** for remote hooks because it requires returning a Python object (a chat model instance or factory callable). It silently becomes a no-op.

All other 16 hooks work over JSON-RPC.

### Error handling

Remote hooks are designed to be resilient. If the hook server is unreachable, times out, or returns an error, sherma:

1. Logs a warning
2. Passes through the original context unchanged
3. Continues normal agent execution

The agent is **never blocked** by a failing hook server.

### Timeout

The default timeout is 30 seconds. You can configure it:

```python
RemoteHookExecutor(url="http://localhost:8000/hooks", timeout=10.0)
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
