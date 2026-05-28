# Plan: Remote Hooks Using MCP Protocol

## Context

Sherma already has a mature hooks system:
- **Local hooks:** Python `HookExecutor` protocol with `BaseHookExecutor` — in-process, direct method calls
- **Remote hooks:** `RemoteHookExecutor` using JSON-RPC 2.0 over HTTP (`sherma/hooks/remote.py`)
- **20 hook types** covering the full agent lifecycle (LLM, tool, agent, skill, node, interrupt, graph, error)
- **Serialization layer** (`sherma/hooks/serialization.py`) that handles non-serializable fields
- **Server SDK:** `HookHandler` base class + ASGI app builders (FastAPI/Starlette)

The question: can/should sherma also support MCP as a transport for remote hooks?

## Reference: Claude Code's MCP Hooks

Claude Code supports hooks that match MCP tool calls via `mcp__<server>__<tool>` naming patterns. This is a different concept — it's about matching *tool invocations* that happen to be MCP tools. Sherma's hooks are lifecycle hooks (before/after LLM call, node enter/exit, etc.), which is a different concern but can use MCP as a transport.

## Design: MCP as a Remote Hook Transport

### Core Idea

Add an `MCPHookExecutor` that uses MCP as the transport protocol for remote hooks — replacing the HTTP/JSON-RPC transport with MCP's tool-call mechanism while keeping the same hook semantics.

An MCP hook server exposes **one tool per hook type** it handles. The tool input is the serialized hook context; the output is the modified context (or `null` for pass-through). This maps 1:1 to the existing JSON-RPC approach.

### Architecture

```
┌─────────────────────────────────────────────────┐
│  Sherma Agent                                    │
│                                                  │
│  HookManager                                     │
│    ├── LocalHookExecutor  (in-process Python)    │
│    ├── RemoteHookExecutor (JSON-RPC over HTTP)   │
│    └── MCPHookExecutor    (MCP tools) ← NEW      │
│           │                                      │
│           │  MCP transport (stdio / SSE / HTTP)  │
│           ▼                                      │
│  ┌──────────────────────────────┐                │
│  │  MCP Hook Server             │                │
│  │  Tools:                      │                │
│  │    hooks/before_llm_call     │                │
│  │    hooks/after_llm_call      │                │
│  │    hooks/node_enter          │                │
│  │    hooks/node_exit           │                │
│  │    ...                       │                │
│  └──────────────────────────────┘                │
└─────────────────────────────────────────────────┘
```

### MCP Tool Convention

Each hook is exposed as an MCP tool with the naming pattern `hooks/{hook_name}`:

| MCP Tool Name | Input Schema | Output |
|---|---|---|
| `hooks/before_llm_call` | `BeforeLLMCallContext` (serialized) | Modified context dict \| `null` |
| `hooks/after_llm_call` | `AfterLLMCallContext` (serialized) | Modified context dict \| `null` |
| `hooks/before_tool_call` | `BeforeToolCallContext` (serialized) | Modified context dict \| `null` |
| `hooks/after_tool_call` | `AfterToolCallContext` (serialized) | Modified context dict \| `null` |
| `hooks/node_enter` | `NodeEnterContext` (serialized) | Modified context dict \| `null` |
| `hooks/node_exit` | `NodeExitContext` (serialized) | Modified context dict \| `null` |
| ... | ... | ... |

The server only registers tools for hooks it actually handles. `MCPHookExecutor` discovers available hooks via `list_tools` at connection time.

### Serialization

Reuse the existing `sherma/hooks/serialization.py` module. The same rules apply:
- Non-serializable fields (`node_context`, `agent`, `chat_model`, `registries`) are stripped on send, re-attached on receive
- Complex fields (`messages`, `response`, `tools`, `tool_calls`) are serialized read-only
- Primitive/dict fields (`state`, `node_name`, `result`, etc.) are fully mutable

### YAML Configuration

```yaml
hooks:
  # Local Python hook (existing)
  - import_path: my_hooks.MyHook

  # JSON-RPC remote hook (existing)
  - url: http://localhost:8000/hooks

  # MCP remote hook — stdio transport (NEW)
  - mcp:
      command: python my_mcp_hook_server.py
      args: ["--config", "hooks.yaml"]

  # MCP remote hook — SSE/streamable HTTP transport (NEW)
  - mcp:
      url: http://localhost:9000/mcp
```

Schema change to `HookDef`:

```python
class MCPHookDef(BaseModel):
    command: str | None = None    # stdio transport
    args: list[str] = []
    url: str | None = None        # SSE / streamable HTTP transport
    env: dict[str, str] = {}      # extra env vars for stdio

    @model_validator(mode="after")
    def _check_one_transport(self):
        if not (bool(self.command) ^ bool(self.url)):
            raise ValueError("Exactly one of 'command' or 'url' must be set")
        return self

class HookDef(BaseModel):
    import_path: str | None = None
    url: str | None = None
    mcp: MCPHookDef | None = None

    @model_validator(mode="after")
    def _check_one_source(self):
        sources = [self.import_path, self.url, self.mcp]
        if sum(s is not None for s in sources) != 1:
            raise ValueError("Exactly one of 'import_path', 'url', or 'mcp' must be set")
        return self
```

## Implementation Steps

### Step 1: `MCPHookExecutor` — Client Side

**File:** `sherma/hooks/mcp_executor.py`

```python
class MCPHookExecutor(BaseHookExecutor):
    """Hook executor that delegates to an MCP server."""

    def __init__(self, client: MCPClient):
        self._client = client
        self._available_hooks: set[str] = set()

    async def connect(self) -> None:
        """Connect to MCP server and discover available hook tools."""
        tools = await self._client.list_tools()
        self._available_hooks = {
            t.name.removeprefix("hooks/")
            for t in tools
            if t.name.startswith("hooks/")
        }

    async def _execute_hook(self, hook_name: str, ctx: Any) -> Any:
        tool_name = f"hooks/{hook_name}"
        if hook_name not in self._available_hooks:
            return None  # Server doesn't handle this hook
        params = serialize_context(ctx)
        result = await self._client.call_tool(tool_name, params)
        if result is None:
            return None
        return deserialize_context(type(ctx), result, ctx)

    async def before_llm_call(self, ctx):
        return await self._execute_hook("before_llm_call", ctx)

    async def after_llm_call(self, ctx):
        return await self._execute_hook("after_llm_call", ctx)

    # ... same pattern for all 20 hooks

    async def on_chat_model_create(self, ctx):
        return None  # Not supported over MCP (non-serializable)
```

**Dependencies:** `mcp` Python SDK (already a dependency for skill MCP loading).

### Step 2: MCP Hook Server SDK — Server Side

**File:** `sherma/hooks/mcp_server.py`

Provide a helper that wraps the existing `HookHandler` interface into an MCP server:

```python
from mcp.server import Server

class MCPHookServer:
    """Wraps a HookHandler as an MCP server."""

    def __init__(self, handler: HookHandler, name: str = "sherma-hooks"):
        self._handler = handler
        self._server = Server(name)
        self._register_tools()

    def _register_tools(self) -> None:
        """Register MCP tools for each hook method that is overridden."""
        for hook_name in HOOK_METHODS:
            method = getattr(self._handler, hook_name)
            if method.__func__ is not getattr(HookHandler, hook_name):
                # Only register tools for overridden methods
                self._register_hook_tool(hook_name, method)

    def _register_hook_tool(self, hook_name: str, method) -> None:
        @self._server.tool(name=f"hooks/{hook_name}")
        async def handle(params: dict) -> dict | None:
            return await method(params)

    async def run_stdio(self) -> None:
        """Run as stdio MCP server."""
        from mcp.server.stdio import stdio_server
        async with stdio_server() as (read, write):
            await self._server.run(read, write)
```

**Usage:**

```python
from sherma.hooks.mcp_server import MCPHookServer
from sherma.hooks.handler import HookHandler

class MyHooks(HookHandler):
    async def before_llm_call(self, params):
        params["system_prompt"] += "\nBe concise."
        return params

server = MCPHookServer(handler=MyHooks())
# Run as stdio
asyncio.run(server.run_stdio())
# Or get the Server instance for custom transport
mcp_server = server.server
```

### Step 3: Schema Update

**File:** `sherma/langgraph/declarative/schema.py`

Add `MCPHookDef` model and update `HookDef` to support the `mcp:` key (as shown above).

### Step 4: Loader Update

**File:** `sherma/langgraph/declarative/loader.py`

Update `populate_hooks` to handle MCP hook definitions:

```python
async def populate_hooks(config: DeclarativeConfig, hook_manager: HookManager) -> None:
    for hook_def in config.hooks:
        if hook_def.mcp:
            executor = await _create_mcp_hook_executor(hook_def.mcp)
            hook_manager.register(executor)
        elif hook_def.url:
            hook_manager.register(RemoteHookExecutor(url=hook_def.url))
        else:
            # import_path handling (existing)
            ...
```

Note: `populate_hooks` becomes `async` because MCP connection + tool discovery requires `await`.

### Step 5: Lifecycle Management

MCP connections (especially stdio) need proper startup/shutdown:

- **Startup:** Connect to MCP server, call `list_tools` to discover hooks
- **Shutdown:** Gracefully close MCP connection

Add lifecycle hooks to `LangGraphAgent`:
```python
async def _start_mcp_hooks(self) -> None:
    """Called during agent initialization."""

async def _stop_mcp_hooks(self) -> None:
    """Called during agent shutdown."""
```

This mirrors how MCP tool connections are managed for skills.

### Step 6: Tests

- **Unit tests:** `tests/hooks/test_mcp_executor.py` — mock MCP client, verify tool discovery and hook dispatch
- **Unit tests:** `tests/hooks/test_mcp_server.py` — verify tool registration, only overridden hooks exposed
- **Integration tests:** `tests/integration/test_mcp_hooks.py` — end-to-end with stdio MCP server
- **Schema tests:** Verify `HookDef` validation with `mcp:` key

### Step 7: Update Documentation and Skill

- Update `docs/hooks.md` — add MCP hooks section alongside existing JSON-RPC section
- Update `skills/sherma/references/` — mirror docs changes
- Update `skills/sherma/SKILL.md` — add MCP hooks to quick reference

## Design Decisions

### Should MCP be the only remote hooks protocol?

**Recommendation: No, keep both JSON-RPC and MCP.**

| | JSON-RPC (existing) | MCP (new) |
|---|---|---|
| **Transport** | HTTP only | stdio, SSE, streamable HTTP |
| **Discovery** | None (all hooks called) | `list_tools` (selective) |
| **Setup complexity** | Low (any HTTP server) | Medium (MCP SDK required) |
| **Ecosystem** | Custom | Standard (Claude Code, etc.) |
| **Best for** | Simple HTTP webhooks | Sidecar processes, ecosystem integration |

Users choose based on their deployment model. JSON-RPC is simpler; MCP is more capable.

### Which hooks are supported over MCP?

All hooks except `on_chat_model_create` (which involves non-serializable Python objects like `ChatModel` instances). This is the same limitation as JSON-RPC remote hooks today.

### Error Handling

Same resilience model as JSON-RPC:
- MCP errors → log warning, return `None` (pass-through)
- Timeout → configurable, default 30s
- Connection failure → log warning, skip hook
- Agent execution is never blocked by a failing hook server

### Auto-Discovery from Skill Cards?

Future consideration: skill cards already have an `mcps:` section for tool loading. We could allow skill cards to also register MCP hook servers:

```yaml
# In skill card
mcps:
  my-hooks:
    url: http://localhost:9000/mcp
    hooks: true  # Also register as hook executor
```

This is out of scope for this task but worth noting as a future enhancement.

## Open Questions

1. **Should MCP hook tools use a namespace prefix?** Proposed: `hooks/before_llm_call`. Alternative: `sherma_hooks.before_llm_call` or just `before_llm_call`.
2. **Should we support MCP notifications (one-way) for read-only hooks?** Some hooks like `node_enter` are often observational only. MCP notifications could reduce latency.
3. **Should we auto-detect hook tools?** Instead of requiring the `hooks/` prefix, should `MCPHookExecutor` match any tool whose name matches a known hook name?

## Plan Revisions

### 2026-05-28: Open questions resolved (from issue #77 comment)

1. **MCP tool prefix:** Use a `hooks` namespace prefix. The user requested `hooks/before_llm_call`, but the MCP spec restricts tool names to `[A-Za-z0-9_.-]+`, so `/` is not valid. Using `.` as the separator instead: `hooks.before_llm_call`. Same namespacing semantics.
2. **MCP notifications:** Not supported. Hooks are always request/response. `MCPHookServer` does not expose notification-based variants.
3. **Auto-detection:** Tools must use the `hooks.` prefix to be considered hook tools. No auto-detection by name alone.

Open Questions section can now be considered resolved.

### 2026-05-28: Implementation refinements

- Adopted **FastMCP** (from the `mcp` Python SDK) instead of the lower-level `Server` class — `FastMCP` is built into the existing dependency and simplifies tool registration plus offers stdio / sse / streamable-http transports out of the box.
- `MCPHookExecutor` accepts `MCPHookTransport` config (stdio command or HTTP URL) and manages connection lifecycle internally via `AsyncExitStack`. The client opens a `ClientSession`, performs `initialize`, then calls `list_tools` to populate the available-hook set.
- `MCPHookServer` builds a `FastMCP` instance, inspects which `HookHandler` methods have been overridden (by comparing function references against the base class), and registers each as a tool named `hooks.<hook_name>`.
- Per-hook errors on the client side log a warning and return `None` (pass-through) — same resilience model as `RemoteHookExecutor`.
