# API Reference

All public symbols are exported from the `sherma` package.

```python
import sherma
```

## Entities

### `EntityBase`

Base class for all registry entities.

```python
class EntityBase(BaseModel):
    id: str
    version: str = "*"
    tenant_id: str = DEFAULT_TENANT_ID  # "default"
```

### `Prompt`

```python
class Prompt(EntityBase):
    instructions: str
```

### `LLM`

```python
class LLM(EntityBase):
    model_name: str
```

### `Tool`

```python
class Tool(EntityBase):
    function: Callable
```

### `Skill`

```python
class Skill(EntityBase):
    front_matter: SkillFrontMatter
    body: Markdown = ""
    scripts: list[Tool] = []
    references: list[Markdown] = []
    assets: list[Any] = []
```

### `SkillFrontMatter`

```python
class SkillFrontMatter(BaseModel):
    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, Any] | None = None
    allowed_tools: list[str] | None = None
```

### `SkillCard`

```python
class SkillCard(EntityBase):
    name: str
    description: str
    base_uri: str
    files: list[str] = []
    mcps: dict[str, MCPServerDef] = {}
    local_tools: dict[str, LocalToolDef] = {}
    extensions: list[SkillExtension] = []
```

### `SkillExtension`

Modeled after the [A2A AgentExtension](https://a2a-protocol.org/latest/specification/#444-agentextension).

```python
class SkillExtension(BaseModel):
    uri: str
    description: str | None = None
    required: bool = False
    params: dict[str, Any] | None = None
```

### `MCPServerDef`

```python
class MCPServerDef(BaseModel):
    id: str
    version: str = "*"
    url: str
    transport: str  # "stdio" | "sse" | "streamable-http"
```

### `LocalToolDef`

```python
class LocalToolDef(BaseModel):
    id: str
    version: str = "*"
    import_path: str
```

## Agents

### `Agent`

Abstract base class for all agents.

```python
class Agent(EntityBase, ABC):
    agent_card: AgentCard | None = None
    input_schema: type[BaseModel] | dict[str, Any] | None = None
    output_schema: type[BaseModel] | dict[str, Any] | None = None

    def send_message(self, request: Message, ...) -> AsyncIterator[UpdateEvent | Message | Task]
    async def cancel_task(self, request: TaskIdParams, ...) -> Task
    async def get_card(self) -> AgentCard | None
```

`input_schema` and `output_schema` accept either a Pydantic model
class or a raw JSON Schema dict. Validation utilities and the A2A
executor dispatch on the value's type, so both forms are treated
identically. Declarative agents populate these from the YAML's
`AgentDef.input_schema` / `AgentDef.output_schema` blocks.

### `LocalAgent`

Agent running in the same process.

### `RemoteAgent`

Proxy for an A2A-compatible remote agent.

### `LangGraphAgent`

Agent backed by a LangGraph compiled state graph.

```python
class LangGraphAgent(Agent):
    hook_manager: HookManager

    def register_hooks(self, executor: HookExecutor) -> None
    async def get_graph(self) -> CompiledStateGraph  # abstract
```

`send_message` and `cancel_task` are auto-implemented. You only implement `get_graph()`.

### `DeclarativeAgent`

Agent defined by YAML and CEL. Extends `LangGraphAgent`.

```python
class DeclarativeAgent(LangGraphAgent):
    yaml_path: str | Path | None = None
    yaml_content: str | None = None
    config: DeclarativeConfig | None = None
    base_path: Path | None = None
    http_async_client: Any | None = None
    hooks: list[HookExecutor] = []
    tenant_id: str = DEFAULT_TENANT_ID  # "default"
    checkpointer: BaseCheckpointSaver = MemorySaver()  # State persistence
```

Provide one of `yaml_path`, `yaml_content`, or `config`.

When `yaml_path` is provided, `base_path` is automatically derived from the YAML file's parent directory. When using `yaml_content` or `config`, set `base_path` explicitly to resolve relative file paths (skill card paths, sub-agent YAML paths).

## Registries

### `RegistryEntry`

```python
class RegistryEntry(BaseModel, Generic[T]):
    id: str
    version: str = "*"
    tenant_id: str = DEFAULT_TENANT_ID  # "default"
    remote: bool = False
    instance: T | None = None
    factory: Callable[[], T | Awaitable[T]] | None = None
    url: str | None = None
    protocol: Protocol | None = None
```

### `Registry`

Abstract base class for all registries.

```python
class Registry(ABC, Generic[T]):
    async def add(entry: RegistryEntry[T]) -> None
    async def update(entry: RegistryEntry[T]) -> None
    async def get(entity_id: str, version: str = "*") -> T
    async def fetch(entry: RegistryEntry[T]) -> T  # abstract
    async def refresh(entry: RegistryEntry[T]) -> None
```

### Typed Registries

- `PromptRegistry` -- `Registry[Prompt]`
- `LLMRegistry` -- `Registry[LLM]`
- `ToolRegistry` -- `Registry[Tool]`
- `SkillRegistry` -- `Registry[Skill]` (skills may have a `skill_card: SkillCard` attribute)
- `AgentRegistry` -- `Registry[Agent]`
- `SkillCardRegistry` -- `Registry[SkillCard]`

### `RegistryBundle`

Container for all per-tenant registry instances.

```python
class RegistryBundle(BaseModel):
    tenant_id: str = DEFAULT_TENANT_ID
    tool_registry: ToolRegistry
    llm_registry: LLMRegistry
    prompt_registry: PromptRegistry
    skill_registry: SkillRegistry
    agent_registry: AgentRegistry
    skill_card_registry: SkillCardRegistry
    chat_models: dict[str, Any]
```

### `TenantRegistryManager`

Manages per-tenant singleton `RegistryBundle` instances.

```python
class TenantRegistryManager:
    def get_bundle(self, tenant_id: str = DEFAULT_TENANT_ID) -> RegistryBundle
    def has_tenant(self, tenant_id: str) -> bool
    def list_tenants(self) -> list[str]
    def remove_tenant(self, tenant_id: str) -> None
```

### `DEFAULT_TENANT_ID`

```python
DEFAULT_TENANT_ID = "default"
```

The default tenant ID used when no tenant is explicitly specified.

## Hooks

### `HookExecutor`

Protocol (interface) for hook executors. All methods are async and return `Context | None`.

### `BaseHookExecutor`

Default implementation with all hooks returning `None`. Subclass and override what you need.

### `CELHookExecutor`

Hook executor whose behaviour is defined by CEL expressions instead of Python code. Implements the `HookExecutor` protocol. Used by the declarative `cel` hook source; can also be constructed directly.

```python
class CELHookExecutor(BaseHookExecutor):
    def __init__(self, spec: dict[str, dict[str, Any]], cel: CelEngine) -> None
```

`spec` maps a hook name to `{"when": <expr?>, "state_updates": {<key>: <expr>}, "set": {<field>: <expr>}}`. Each hook context's fields are exposed at the top level of the activation (graph state under `state`); `state_updates` merge into the context's `result` when present (persisting into graph state) else into `state`; `set` assigns other writable scalar/dict fields. `on_chat_model_create` is unsupported. See [CEL Hooks](hooks.md#cel-hooks).

The module also exposes `writable_fields(hook_name)`, `hook_supports_state_updates(hook_name)`, and the `CEL_CAPABLE_HOOKS` frozenset, used to validate `cel` hook specs.

### `RemoteHookExecutor`

Hook executor that delegates to a remote JSON-RPC 2.0 server. Implements the `HookExecutor` protocol.

```python
class RemoteHookExecutor(BaseHookExecutor):
    def __init__(self, url: str, timeout: float = 30.0) -> None
```

The `on_chat_model_create` hook is a no-op (cannot return Python objects over JSON-RPC). On any network or protocol error, logs a warning and passes through.

Uses the shared HTTP client returned by `sherma.http.get_http_client()`, so headers, auth, transport, event hooks, and timeouts configured there are applied to remote hook calls. The `timeout` constructor argument is applied per-request and takes precedence over the shared client's timeout.

### `MCPHookTransport`

Connection parameters for an MCP hook server. Provide exactly one of `command` (stdio) or `url` (HTTP).

```python
@dataclass
class MCPHookTransport:
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    transport: Literal["streamable_http", "sse"] = "streamable_http"
    headers: dict[str, str] = field(default_factory=dict)
```

### `MCPHookExecutor`

Hook executor that delegates to an MCP server. Each hook is invoked as an MCP tool named `hooks.<hook_name>`. Implements the `HookExecutor` protocol.

```python
class MCPHookExecutor(BaseHookExecutor):
    def __init__(self, transport: MCPHookTransport) -> None
    async def connect(self) -> None          # open connection + discover hook tools
    async def close(self) -> None            # close connection (idempotent)
    @property
    def available_hooks(self) -> set[str]    # hook names discovered on the server
```

Call `connect()` before use and `close()` when done. The `on_chat_model_create` hook is a no-op. On any error, logs a warning and passes through. Hooks not registered on the server are skipped without any IPC.

For `streamable_http` and `sse` transports, the MCP SDK's httpx client is built by `sherma.http.mcp_http_client_factory()` so that event hooks, auth, transport, and default headers from the shared HTTP client flow through. MCP-supplied headers and auth win over shared-client defaults on conflict. The stdio transport does not use HTTP.

### `MCPHookServer`

Wraps a `HookExecutor` as an MCP server. Only overridden hook methods are exposed as MCP tools named `hooks.<hook_name>`. Hooks are authored with the same typed context dataclasses as in-process hooks; the server reconstructs the context from the incoming params and serializes the returned context back.

```python
class MCPHookServer:
    def __init__(self, executor: HookExecutor, name: str = "sherma-hooks",
                 instructions: str | None = None) -> None
    @property
    def server(self) -> FastMCP             # underlying FastMCP instance
    @property
    def registered_hooks(self) -> set[str]  # hooks exposed as tools
    def run_stdio(self) -> None
    def run_streamable_http(self, mount_path: str | None = None) -> None
    def run_sse(self, mount_path: str | None = None) -> None
```

### Authoring a hook server

Remote hook servers (both JSON-RPC and MCP) are authored as `BaseHookExecutor` subclasses with the typed context dataclasses — the same interface as in-process hooks. On the server side, non-serializable fields (`node_context`, `agent`, `registries`) are `None` and complex fields (`messages`, `tools`, `response`) carry their serialized form. `on_chat_model_create` is never dispatched remotely.

### `HookFastAPIApplication`

Builds a FastAPI application from a `HookExecutor`.

```python
class HookFastAPIApplication:
    def __init__(self, executor: HookExecutor) -> None
    def build(self, rpc_url: str = "/hooks", **kwargs) -> FastAPI
    def add_routes_to_app(self, app: FastAPI, rpc_url: str = "/hooks") -> None
```

### `HookStarletteApplication`

Builds a Starlette application from a `HookExecutor`.

```python
class HookStarletteApplication:
    def __init__(self, executor: HookExecutor) -> None
    def build(self, rpc_url: str = "/hooks", **kwargs) -> Starlette
    def add_routes_to_app(self, app: Starlette, rpc_url: str = "/hooks") -> None
    def routes(self, rpc_url: str = "/hooks") -> list[Route]
```

### `HookManager`

```python
class HookManager:
    def register(self, executor: HookExecutor) -> None
    async def run_hook(self, hook_name: str, ctx: T) -> T
```

### `HookType`

```python
class HookType(Enum):
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_AGENT_CALL = "before_agent_call"
    AFTER_AGENT_CALL = "after_agent_call"
    BEFORE_SKILL_LOAD = "before_skill_load"
    AFTER_SKILL_LOAD = "after_skill_load"
    NODE_ENTER = "node_enter"
    NODE_EXECUTE = "node_execute"
    NODE_EXIT = "node_exit"
    BEFORE_INTERRUPT = "before_interrupt"
    AFTER_INTERRUPT = "after_interrupt"
    ON_CHAT_MODEL_CREATE = "on_chat_model_create"
    BEFORE_GRAPH_INVOKE = "before_graph_invoke"
    AFTER_GRAPH_INVOKE = "after_graph_invoke"
    ON_NODE_ERROR = "on_node_error"
    ON_ERROR = "on_error"
```

### Hook Context Types

Imported from `sherma.hooks.types`:

- `BeforeLLMCallContext`
- `AfterLLMCallContext`
- `BeforeToolCallContext`
- `AfterToolCallContext`
- `BeforeAgentCallContext`
- `AfterAgentCallContext`
- `BeforeSkillLoadContext`
- `AfterSkillLoadContext`
- `NodeEnterContext`
- `NodeExecuteContext`
- `NodeExitContext`
- `BeforeInterruptContext`
- `AfterInterruptContext`
- `ChatModelCreateContext`
- `GraphInvokeContext`
- `AfterGraphInvokeContext`
- `OnNodeErrorContext`
- `OnErrorContext`

## Declarative Config

### `DeclarativeConfig`

Top-level YAML schema model.

```python
class DeclarativeConfig(BaseModel):
    manifest_version: int             # Required: schema version (currently 1)
    agents: dict[str, AgentDef] = {}
    llms: list[LLMDef] = []
    tools: list[ToolDef] = []
    prompts: list[PromptDef] = []
    skills: list[SkillDef] = []
    hooks: list[HookDef] = []
    sub_agents: list[SubAgentDef] = []
    mcp_servers: list[MCPServerDef] = []
    default_llm: RegistryRef | None = None
    checkpointer: CheckpointerDef | None = None
```

`manifest_version` is a required integer that tracks which version of the declarative agent schema the config uses. The current version is **1**. This enables the runtime to handle configs with different schema versions.

`default_llm` is an optional `RegistryRef` that `call_llm` nodes inherit when they omit the step-level `llm` field. A step-level `llm` always takes precedence.

### `PromptDef`

```python
class PromptDef(BaseModel):
    id: str
    version: str = "*"
    instructions: str | None = None       # Inline prompt body
    instructions_path: str | None = None  # Path to a file containing the prompt body
```

Exactly one of `instructions` or `instructions_path` must be provided.
Relative `instructions_path` values are resolved against the YAML's
`base_path` (see `populate_registries`); absolute paths are used as-is.
The file contents are read as UTF-8 text and used verbatim as the prompt
instructions.

### `HookDef`

```python
class HookDef(BaseModel):
    import_path: str | None = None            # Local Python hook executor
    url: str | None = None                    # Remote JSON-RPC hook server
    mcp: MCPHookDef | None = None             # MCP-based hook server
    cel: dict[str, CELHookSpec] | None = None # Inline CEL-expression hooks
```

Exactly one of `import_path`, `url`, `mcp`, or `cel` must be provided.

```python
class CELHookSpec(BaseModel):
    when: str | None = None                  # optional CEL boolean guard
    state_updates: dict[str, str] = {}       # state_key -> CEL expression
    set: dict[str, str] = {}                 # field -> CEL expression
```

A `cel` map is keyed by hook name. Each spec needs at least one of `state_updates` / `set`. Hook names are validated, `on_chat_model_create` is rejected, `state_updates` require a `result`/`state`-bearing context, and `set` keys must be writable fields. See [CEL Hooks](hooks.md#cel-hooks).

### `CheckpointerDef`

```python
class CheckpointerDef(BaseModel):
    type: Literal["memory"] = "memory"
```

### `MCPServerDef` (declarative YAML)

Top-level `mcp_servers:` entry — distinct from the same-named class
under `sherma.entities.skill_card`, which models MCP servers embedded
in a skill card. This one lives in
`sherma.langgraph.declarative.schema`.

```python
class MCPServerDef(BaseModel):
    id: str
    version: str = "*"
    transport: Literal["streamable_http", "sse", "stdio"] = "streamable_http"
    # streamable_http / sse
    url: str | None = None
    headers: dict[str, str] = {}
    # stdio
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    # Optional renaming to avoid name collisions across servers
    tool_prefix: str | None = None
```

For HTTP-based transports, set `url` (and optionally `headers`).
For `stdio`, set `command` (and optionally `args` / `env`).
At config-load time sherma connects to each declared server,
lists its tools, and registers them in the tool registry — making
them usable from `call_llm` nodes via `tools:` or
`use_tools_from_registry: true`. When `tool_prefix` is set, each
tool is registered as `<tool_prefix><tool_name>`.

### `load_declarative_config`

```python
def load_declarative_config(
    yaml_path: str | Path | None = None,
    yaml_content: str | None = None,
) -> DeclarativeConfig
```

Before the YAML data is validated, every string value is passed
through environment-variable interpolation: `${VAR}` is replaced
with `os.environ["VAR"]`, `${VAR:-default}` falls back to `default`
when the variable is unset, and `$$` becomes a literal `$`. Only
`UPPERCASE_WITH_UNDERSCORES` names are matched, so lowercase
placeholders intended for the CEL `template()` function (e.g.
`${available_skills}`) pass through untouched. Missing required
variables raise `DeclarativeConfigError` listing all unresolved
names.

## Schema Utilities

### Constants

```python
SCHEMA_INPUT_URI = "urn:sherma:schema:input"
SCHEMA_OUTPUT_URI = "urn:sherma:schema:output"
```

### Functions

```python
def validate_data(data: dict, schema_model: type[BaseModel]) -> BaseModel
def validate_json_schema_data(data: dict, schema: dict) -> None
def validate_against_schema(data: dict, schema: type[BaseModel] | dict) -> None
def schema_to_extension(uri: str, schema: type[BaseModel] | dict) -> AgentExtension
def make_schema_data_part(data: dict, schema_uri: str, *, extra_metadata=None) -> Part

def create_agent_input_as_message_part(data, schema_uri, *, role=Role.user, ...) -> Message
def create_agent_output_as_message_part(data, schema_uri, *, role=Role.agent, ...) -> Message
def get_agent_input_from_message_part(message: Message, schema_model) -> BaseModel
def get_agent_output_from_message_part(message: Message, schema_model) -> BaseModel
```

## LangGraph Utilities

### `combine_ai_messages`

```python
from sherma.langgraph.agent import combine_ai_messages

def combine_ai_messages(messages: list[AIMessage]) -> AIMessage
```

Merges multiple `AIMessage` instances into one by concatenating their content into list-form. Collapses to a plain string when the result contains exactly one text block.

### `LazyChatModel`

```python
from sherma.langgraph.declarative.loader import LazyChatModel

proxy = LazyChatModel(factory=lambda: ChatOpenAI(model="gpt-4o"))
```

A transparent proxy that defers chat model construction until first attribute access. Used internally when `on_chat_model_create` hooks set `chat_model` to a callable factory. All attribute access and method calls are forwarded to the real model after construction.

## Skill Tools

```python
def create_skill_tools(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    hook_manager: HookManager | None = None,
) -> list[BaseTool]
```

Returns: `list_skills`, `load_skill_md`, `unload_skill`, `list_skill_resources`, `load_skill_resource`, `list_skill_assets`, `load_skill_asset`.

## Types

```python
Markdown = str  # Type alias

class Protocol(StrEnum):
    A2A = "a2a"
    MCP = "mcp"
    CUSTOM = "custom"

class EntityType(StrEnum):
    PROMPT = "prompt"
    LLM = "llm"
    TOOL = "tool"
    SKILL = "skill"
    AGENT = "agent"
```

## HTTP Client

`sherma.http` exposes a single shared `httpx.AsyncClient` cached in a `ContextVar`. All built-in HTTP traffic — LLM provider calls, A2A remote agents, skill/prompt/skill-card registries, remote hooks (JSON-RPC), and MCP hooks (`streamable_http` / `sse`) — flows through the same configuration, so users can customize headers, auth, event hooks, transport (proxies, retries, mocks), and timeouts in one place.

### `get_http_client`

```python
HttpClientFactory = Callable[[], httpx.AsyncClient]

async def get_http_client(
    client: httpx.AsyncClient | HttpClientFactory | None = None,
) -> httpx.AsyncClient
```

Returns the shared client. Pass an `httpx.AsyncClient` instance or a zero-arg factory to seed the context (typically once, at agent setup). Subsequent calls with no argument return the cached client.

### `mcp_http_client_factory`

```python
class McpHttpClientFactory(Protocol):
    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient: ...

async def mcp_http_client_factory() -> McpHttpClientFactory
```

Builds the `httpx_client_factory` consumed by the MCP SDK's `streamablehttp_client` and `sse_client`. The MCP SDK opens its HTTP transports inside `async with` and closes the returned client, so the factory hands back a **fresh** `httpx.AsyncClient` per MCP connection, inheriting configuration from the shared client:

| Attribute | Behavior |
| --- | --- |
| `event_hooks` | copied |
| `transport` | reused (proxies, retries, mocks flow through) |
| default headers | merged; **MCP-supplied headers win on conflict** |
| `auth` | inherited only if MCP did not supply one |
| `timeout` | MCP's value wins; falls back to the shared client's |

Used internally by `MCPHookExecutor`; agent authors rarely call it directly.

## Exceptions

All exceptions inherit from `ShermaError`:

| Exception | Description |
| --- | --- |
| `ShermaError` | Base exception |
| `EntityNotFoundError` | Entity not in registry |
| `VersionNotFoundError` | No matching version found |
| `RegistryError` | General registry error |
| `RemoteEntityError` | Failed to fetch remote entity |
| `DeclarativeConfigError` | Invalid YAML config |
| `GraphConstructionError` | Error building graph from config |
| `CelEvaluationError` | CEL expression evaluation failed |
| `SchemaValidationError` | Input/output schema validation failed |
