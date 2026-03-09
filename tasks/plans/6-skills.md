# Plan: Skills Support (Task 6)

## Context

Sherma needs skills support following the agentskills.io specification. Skills enable progressive disclosure: agents see metadata at startup, load full instructions on activation, and access resources/assets on demand. An MCP extension allows skills to attach MCP tools when loaded.

## Phase 1: SkillCard Data Model

### New file: `sherma/entities/skill_card.py`

```python
class MCPServerDef(BaseModel):
    id: str
    version: str = "*"
    url: str
    transport: str  # "stdio" | "sse" | "streamable-http"

class LocalToolDef(BaseModel):
    id: str
    version: str = "*"
    import_path: str  # e.g. "examples.tools.get_weather"

class SkillCard(EntityBase):
    name: str
    description: str
    base_uri: str  # filesystem path or HTTP URL
    files: list[str] = Field(default_factory=list)
    mcps: dict[str, MCPServerDef] = Field(default_factory=dict)
    local_tools: dict[str, LocalToolDef] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)  # future extensibility
```

- Inherits `EntityBase` (provides `id`, `version`) to satisfy `Registry[T]` constraint
- `base_uri` determines local vs remote: HTTP prefix = remote, else local filesystem
- `files` contains relative paths like `"SKILL.md"`, `"assets/x.png"`, `"references/y.md"`
- `mcps` maps server name -> MCP server definition (url + transport)
- `local_tools` maps tool name -> local tool definition (import_path), same mechanism as declarative agent tool imports
- `extensions` reserved for future extensibility

Example skill-card.json:
```json
{
    "id": "my-skill",
    "version": "1.0.0",
    "name": "My Skill",
    "description": "Does something useful",
    "base_uri": "/path/to/skill",
    "files": ["SKILL.md", "assets/diagram.png", "references/api.md"],
    "mcps": {
        "my-mcp-server": {
            "id": "my-mcp-server",
            "version": "1.0.0",
            "url": "http://localhost:8080",
            "transport": "streamable-http"
        }
    },
    "local_tools": {
        "my-local-tool": {
            "id": "my-local-tool",
            "version": "1.0.0",
            "import_path": "my_module.tools.my_tool"
        }
    }
}
```

### Modify: `sherma/entities/__init__.py`
- Export `SkillCard`, `MCPServerDef`, `LocalToolDef`

## Phase 2: SkillCard Registry

### New file: `sherma/registry/skill_card.py`

```python
class SkillCardRegistry(Registry[SkillCard]):
    async def fetch(self, entry: RegistryEntry[SkillCard]) -> SkillCard:
        # GET url, parse JSON -> SkillCard
```

Follows exact same pattern as `SkillRegistry.fetch()`.

### Modify: `sherma/registry/__init__.py`
- Export `SkillCardRegistry`

## Phase 3: Skill Resolver

### New file: `sherma/skills/__init__.py` (empty)
### New file: `sherma/skills/resolver.py`

```python
class SkillResolver:
    def __init__(self, skill_card: SkillCard): ...

    def is_remote(self) -> bool: ...
    def resolve_path(self, relative_path: str) -> str: ...
    async def load_file(self, relative_path: str) -> str: ...
    def list_files_by_prefix(self, prefix: str) -> list[str]: ...
```

- Local: reads from `Path(base_uri) / relative_path`
- Remote: GET `base_uri + "/" + relative_path` via `get_http_client()`

## Phase 4: LangGraph Skill Tools (Progressive Disclosure)

### New file: `sherma/langgraph/skill_tools.py`

Factory function `create_skill_tools(skill_card_registry, skill_registry, tool_registry)` returns `list[BaseTool]`:

1. **`list_skills()`** - Returns `list[{id, version, name, description}]` from all skill cards (~100 tokens each)
2. **`load_skill_md(skill_id, version)`** - Resolves SKILL.md via SkillResolver, parses with `_parse_skill_md`, stores in SkillRegistry, returns markdown body
3. **`list_skill_resources(skill_id, version)`** - Filters `skill_card.files` for `references/` prefix
4. **`load_skill_resource(skill_id, version, resource_path)`** - Loads specific reference file content
5. **`list_skill_assets(skill_id, version)`** - Filters `skill_card.files` for `assets/` prefix
6. **`load_skill_asset(skill_id, version, asset_path)`** - Loads specific asset content

All tools are `@tool`-decorated async functions (closures over registries).

### Modify: `sherma/langgraph/__init__.py`
- Export `create_skill_tools`

## Phase 5: Tool Loading from Skills

### New file: `sherma/skills/mcp.py`

```python
async def load_mcp_tools_from_skill(skill_card: SkillCard) -> list[BaseTool]:
    # Iterate skill_card.mcps entries
    # Use langchain-mcp-adapters MultiServerMCPClient to connect by url + transport
    # Return list of LangChain BaseTool instances
```

### New file: `sherma/skills/local_tools.py`

```python
def load_local_tools_from_skill(skill_card: SkillCard) -> list[BaseTool]:
    # Iterate skill_card.local_tools entries
    # Use import_tool() from sherma/langgraph/declarative/loader.py (same mechanism)
    # Return list of LangChain BaseTool instances
```

Integration: When `load_skill_md` is called:
1. Load MCP tools from `skill_card.mcps` and register them in `ToolRegistry`
2. Load local tools from `skill_card.local_tools` and register them in `ToolRegistry`
3. This makes all skill tools available to the agent after skill activation

### Modify: `pyproject.toml`
- Add `langchain-mcp-adapters` dependency

## Phase 6: Declarative Loader Integration

### Modify: `sherma/langgraph/declarative/schema.py`
- Add `skill_card_path: str | None = None` to `SkillDef`

### Modify: `sherma/langgraph/declarative/loader.py`
- Add `SkillCardRegistry` to `RegistryBundle`
- Add skill population in `populate_registries()`:
  - If `skill_card_path`: load local JSON -> register as local SkillCard
  - If `url`: register as remote (fetched on demand)

## Phase 7: Exports

### Modify: `sherma/__init__.py`
- Export `SkillCard`, `MCPServerDef`, `LocalToolDef`, `SkillCardRegistry`, `create_skill_tools`

## Phase 8: Tests

### New test files:
- `tests/entities/test_skill_card.py` - SkillCard model validation, MCPServerDef/LocalToolDef parsing
- `tests/registry/test_skill_card.py` - Local add/get, remote fetch (mock httpx)
- `tests/skills/__init__.py` (empty)
- `tests/skills/test_resolver.py` - Local/remote file resolution, prefix filtering
- `tests/skills/test_mcp.py` - MCP tool loading (mock MCP server)
- `tests/langgraph/test_skill_tools.py` - All 6 tool functions with mocked registries

### Modify:
- `tests/langgraph/declarative/test_loader.py` - Skill population from YAML

## Verification

```bash
uv sync                              # Install new dependency
uv run pytest tests/entities/test_skill_card.py
uv run pytest tests/registry/test_skill_card.py
uv run pytest tests/skills/
uv run pytest tests/langgraph/test_skill_tools.py
uv run pytest                        # Full suite
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

## Key Files Reference

| Existing File | Purpose |
|---|---|
| `sherma/entities/skill.py` | Existing Skill entity (body + frontmatter) - reuse `_parse_skill_md` from registry |
| `sherma/registry/skill.py` | Existing SkillRegistry + `_parse_skill_md` parser - reuse for SKILL.md parsing |
| `sherma/registry/base.py` | `Registry[T]`, `RegistryEntry[T]` base classes |
| `sherma/langgraph/tools.py` | `from_langgraph_tool`, `to_langgraph_tool` converters |
| `sherma/http.py` | `get_http_client()` for async HTTP |
| `sherma/langgraph/declarative/loader.py` | `RegistryBundle`, `populate_registries()` |
| `sherma/langgraph/declarative/schema.py` | `SkillDef`, `DeclarativeConfig` |
