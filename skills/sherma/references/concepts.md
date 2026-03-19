# Core Concepts

sherma is built on a small set of composable primitives. Every agent -- whether programmatic or declarative -- is assembled from these building blocks.

> **Note**: While `DeclarativeAgent` is the fastest way to get started, every utility in sherma (registries, tool wrapping, hooks, message converters, schema helpers) works independently. If you are building agents directly with `LangGraphAgent` or plain LangGraph, you can use these as standalone building blocks without writing any YAML.

## Entities

An **entity** is any named, versioned object used to build an agent. All entities extend `EntityBase`:

```python
class EntityBase(BaseModel):
    id: str
    version: str = "*"
    tenant_id: str = DEFAULT_TENANT_ID  # "default"
```

sherma defines five entity types:

| Entity | Purpose | Key Attributes |
| --- | --- | --- |
| **Prompt** | System or user instructions | `instructions: str` |
| **LLM** | Model reference | `model_name: str` |
| **Tool** | Callable function | `function: Callable` |
| **Skill** | A packaged capability with docs, tools, and assets | `front_matter`, `body`, `scripts`, `references`, `assets` |
| **Agent** | A local or remote agent | `agent_card`, `input_schema`, `output_schema` |

The `EntityType` enum tracks these:

```python
class EntityType(StrEnum):
    PROMPT = "prompt"
    LLM = "llm"
    TOOL = "tool"
    SKILL = "skill"
    AGENT = "agent"
```

## Registry

Every entity is stored in and retrieved from a **registry**. Registries provide a consistent interface for resolving entities by ID and version, regardless of whether the entity is local, factory-constructed, or fetched remotely.

### RegistryEntry

Each entry in a registry wraps an entity with resolution metadata:

```python
class RegistryEntry(BaseModel, Generic[T]):
    id: str
    version: str = "*"
    tenant_id: str = DEFAULT_TENANT_ID     # "default"
    remote: bool = False
    instance: T | None = None              # Direct instance
    factory: Callable[[], T | Awaitable[T]] | None = None  # Lazy factory
    url: str | None = None                 # Remote URL
    protocol: Protocol | None = None       # a2a, mcp, or custom
```

An entry can be resolved in three ways (tried in order):
1. **Instance** -- return the stored object directly
2. **Factory** -- call the factory (sync or async) and cache the result
3. **Remote fetch** -- call the registry's `fetch()` method using the URL and protocol

### Registry Interface

All registries share the same async interface:

```python
class Registry(ABC, Generic[T]):
    async def add(entry: RegistryEntry[T]) -> None
    async def update(entry: RegistryEntry[T]) -> None
    async def get(entity_id: str, version: str = "*") -> T
    async def fetch(entry: RegistryEntry[T]) -> T       # abstract
    async def refresh(entry: RegistryEntry[T]) -> None
```

Each entity type has its own registry subclass: `PromptRegistry`, `LLMRegistry`, `ToolRegistry`, `SkillRegistry`, and `AgentRegistry`.

### Versioning

Versions follow [semver](https://semver.org/). When requesting an entity, you can use:

- `"1.0.0"` -- exact match
- `"1.*"` -- latest minor/patch for major version 1
- `"*"` -- latest version available (default)

The version resolver (`sherma.version.find_best_match`) selects the best matching version from all registered versions.

## Multi-Tenancy

sherma supports per-tenant isolation through `TenantRegistryManager`. Each tenant gets its own `RegistryBundle` -- a container holding independent registry instances for all entity types. Entities registered for one tenant are never accessible from another.

### TenantRegistryManager

```python
from sherma import TenantRegistryManager

manager = TenantRegistryManager()

# Get or create a tenant's registries (singleton per tenant_id)
bundle = manager.get_bundle("acme-corp")
await bundle.tool_registry.add(...)

# Default tenant is "default" -- backward compatible with non-tenant code
default_bundle = manager.get_bundle()  # tenant_id="default"
```

### RegistryBundle

Each tenant's `RegistryBundle` contains independent instances of every registry type:

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

### DeclarativeAgent with Tenants

Pass `tenant_id` when creating a `DeclarativeAgent` to scope it to a specific tenant:

```python
agent = DeclarativeAgent(
    id="weather-agent",
    version="1.0.0",
    yaml_path="weather-agent.yaml",
    tenant_id="acme-corp",
)
```

All code that omits `tenant_id` uses `DEFAULT_TENANT_ID = "default"`, so existing agents continue to work without changes.

## Protocols

sherma recognizes three protocols for remote entities:

```python
class Protocol(StrEnum):
    A2A = "a2a"     # Agent-to-Agent protocol (for agents)
    MCP = "mcp"     # Model Context Protocol (for tools)
    CUSTOM = "custom"  # HTTP GET (for prompts, skills)
```

## Agents

### Agent Base Class

All agents extend the `Agent` abstract class, which mirrors the [A2A client interface](https://github.com/a2aproject/a2a-python):

```python
class Agent(EntityBase, ABC):
    agent_card: AgentCard | None = None
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None

    def send_message(self, request: Message, ...) -> AsyncIterator[UpdateEvent | Message | Task]
    async def cancel_task(self, request: TaskIdParams, ...) -> Task
    async def get_card(self) -> AgentCard | None
```

When `input_schema` or `output_schema` is set, `get_card()` automatically injects the JSON Schema as A2A extensions on the agent card.

### LocalAgent vs RemoteAgent

- **LocalAgent**: An agent running in the same process. You subclass it and implement `send_message` and `cancel_task` directly.
- **RemoteAgent**: A proxy for an agent running elsewhere. Communicates via the A2A protocol using the A2A Python SDK's client.

### LangGraphAgent

`LangGraphAgent` extends `Agent` with automatic LangGraph integration:

```python
class LangGraphAgent(Agent):
    hook_manager: HookManager

    async def get_graph(self) -> CompiledStateGraph  # abstract -- you implement this

    # send_message and cancel_task are auto-implemented:
    # - Converts A2A messages to LangGraph format
    # - Invokes the graph
    # - Converts the response back to A2A
    # - Handles interrupts (input-required state)
```

You only implement `get_graph()`. The framework handles A2A message conversion, graph invocation, and interrupt handling.

### DeclarativeAgent

`DeclarativeAgent` extends `LangGraphAgent`. Instead of implementing `get_graph()` in Python, you provide a YAML file. The graph, registries, and nodes are all built automatically from the YAML config.

See [Declarative Agents](declarative-agents.md) for the full YAML reference.

### Agent-as-Tool

Any agent can be wrapped as a LangGraph tool using `agent_to_langgraph_tool()`. This is the foundation for multi-agent orchestration -- a supervisor agent's LLM can invoke sub-agents through standard tool calling. Declarative agents support this natively via the `sub_agents` config and `use_sub_agents_as_tools` option (set to `true`/`all` for all sub-agents, or a list of `id`/`version` refs for a specific subset).

See [Multi-Agent](multi-agent.md) for the full guide.

## Message Conversion

sherma provides lossless bidirectional conversion between A2A and LangGraph message formats:

- `a2a_to_langgraph(message)` -- A2A `Message` to LangGraph `BaseMessage` list
- `langgraph_to_a2a(message)` -- LangGraph `BaseMessage` to A2A `Message`

Metadata (message ID, task ID, context ID) is preserved in `additional_kwargs` during the round-trip.

## HTTP Client Management

For production use, sherma accepts an `httpx.AsyncClient` (or a factory returning one) for outbound network calls. This allows you to customize headers, timeouts, TLS settings, and connection pooling. If not provided, sherma uses a shared client per request context via `ContextVar`.

## Logging

sherma uses Python's standard `logging` module. All loggers are namespaced under `sherma.*`. Configure them from your application as you would any Python logger.
