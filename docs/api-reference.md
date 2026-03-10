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
    extensions: dict[str, Any] = {}
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
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None

    def send_message(self, request: Message, ...) -> AsyncIterator[UpdateEvent | Message | Task]
    async def cancel_task(self, request: TaskIdParams, ...) -> Task
    async def get_card(self) -> AgentCard | None
```

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
    http_async_client: Any | None = None
    hooks: list[HookExecutor] = []
    tenant_id: str = DEFAULT_TENANT_ID  # "default"
```

Provide one of `yaml_path`, `yaml_content`, or `config`.

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
    NODE_EXIT = "node_exit"
    BEFORE_INTERRUPT = "before_interrupt"
    AFTER_INTERRUPT = "after_interrupt"
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
- `NodeExitContext`
- `BeforeInterruptContext`
- `AfterInterruptContext`

## Declarative Config

### `DeclarativeConfig`

Top-level YAML schema model.

```python
class DeclarativeConfig(BaseModel):
    agents: dict[str, AgentDef] = {}
    llms: list[LLMDef] = []
    tools: list[ToolDef] = []
    prompts: list[PromptDef] = []
    skills: list[SkillDef] = []
    hooks: list[HookDef] = []
```

### `load_declarative_config`

```python
def load_declarative_config(
    yaml_path: str | Path | None = None,
    yaml_content: str | None = None,
) -> DeclarativeConfig
```

## Schema Utilities

### Constants

```python
SCHEMA_INPUT_URI = "urn:sherma:schema:input"
SCHEMA_OUTPUT_URI = "urn:sherma:schema:output"
```

### Functions

```python
def validate_data(data: dict, schema_model: type[BaseModel]) -> BaseModel
def schema_to_extension(uri: str, schema_model: type[BaseModel]) -> AgentExtension
def make_schema_data_part(data: dict, schema_uri: str, *, extra_metadata=None) -> Part

def create_agent_input_as_message_part(data, schema_uri, *, role=Role.user, ...) -> Message
def create_agent_output_as_message_part(data, schema_uri, *, role=Role.agent, ...) -> Message
def get_agent_input_from_message_part(message: Message, schema_model) -> BaseModel
def get_agent_output_from_message_part(message: Message, schema_model) -> BaseModel
```

## Skill Tools

```python
def create_skill_tools(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    hook_manager: HookManager | None = None,
) -> list[BaseTool]
```

Returns: `list_skills`, `load_skill_md`, `list_skill_resources`, `load_skill_resource`, `list_skill_assets`, `load_skill_asset`.

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
