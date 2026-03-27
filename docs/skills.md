# Skills

sherma implements the [Agent Skills](https://agentskills.io/) specification for progressive skill disclosure. Skills are packaged capabilities -- documentation, tools, references, and assets -- that agents discover and load on demand.

## How Skills Work

The skill lifecycle follows the progressive disclosure pattern:

1. **List** -- The LLM calls `list_skills` to see what skills are available (names and descriptions only)
2. **Load** -- The LLM calls `load_skill_md` to read the full skill documentation and activate its tools
3. **Execute** -- The LLM uses the loaded tools to accomplish the task

This lets agents start with a lightweight catalog and only load what they need, keeping context windows efficient.

## Skill Card

A **skill card** (`skill-card.json`) is the discovery manifest for a skill, analogous to an A2A agent card. It declares metadata, file listings, and tool definitions.

```json
{
    "id": "weather",
    "version": "1.0.0",
    "name": "Weather Lookup",
    "description": "Get current weather conditions for any city worldwide.",
    "base_uri": ".",
    "files": [
        "SKILL.md",
        "references/open-meteo-api.md",
        "assets/weather-codes.md"
    ],
    "mcps": {},
    "local_tools": {
        "get_weather": {
            "id": "get_weather",
            "version": "1.0.0",
            "import_path": "my_tools.get_weather"
        }
    },
    "extensions": [
        {
            "uri": "urn:skill:tools:local",
            "description": "Python tool references loaded via import_path"
        }
    ]
}
```

| Field | Description |
| --- | --- |
| `id`, `version` | Unique identifier and semver version |
| `name`, `description` | Human-readable metadata (shown in `list_skills`) |
| `base_uri` | Base path or URL for resolving files |
| `files` | List of files accessible under `base_uri` |
| `mcps` | MCP server definitions for remote tool execution |
| `local_tools` | Python tool references loaded via `import_path` |
| `extensions` | Array of extension declarations (see [A2A AgentExtension](https://a2a-protocol.org/latest/specification/#444-agentextension)). Each entry has a `uri` (required), `description`, `required`, and `params` |

## SKILL.md

The main skill documentation file uses markdown with YAML frontmatter:

```markdown
---
name: Weather Lookup
description: Get current weather conditions for any city worldwide.
license: MIT
---
# Weather Lookup Skill

Use the `get_weather` tool to retrieve current weather for a given city.

## Usage

Call `get_weather(city="<city name>")`. The tool returns a JSON string with:
- Location name and country
- Temperature
- Wind speed and direction
```

When `load_skill_md` is called, sherma:
1. Reads and parses the SKILL.md file
2. Stores the parsed skill in the `SkillRegistry`
3. Loads and registers any MCP tools defined in the skill card
4. Loads and registers any local tools defined in the skill card

## Tool Types

### Local Tools

Local tools are Python functions referenced by import path:

```json
"local_tools": {
    "get_weather": {
        "id": "get_weather",
        "version": "1.0.0",
        "import_path": "my_package.tools.get_weather"
    }
}
```

The import path should point to a `@tool`-decorated LangChain function.

### MCP Tools

MCP (Model Context Protocol) tools connect to remote tool servers:

```json
"mcps": {
    "my-mcp-server": {
        "id": "my-mcp-server",
        "version": "1.0.0",
        "url": "https://mcp.example.com",
        "transport": "streamable-http"
    }
}
```

Supported transports: `stdio`, `sse`, `streamable-http`. sherma uses the [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) library to convert MCP tools to LangChain tools.

## Skill Tools

When skills are declared in a YAML config, sherma creates six LangGraph tools for the LLM:

| Tool | Description |
| --- | --- |
| `list_skills()` | List all available skills with id, version, name, description |
| `load_skill_md(skill_id, version)` | Load SKILL.md and register the skill's tools |
| `list_skill_resources(skill_id, version)` | List reference files in the skill |
| `load_skill_resource(skill_id, resource_path, version)` | Load a specific reference file |
| `list_skill_assets(skill_id, version)` | List asset files in the skill |
| `load_skill_asset(skill_id, asset_path, version)` | Load a specific asset file |

These tools are created by `create_skill_tools()` and registered automatically when skills are present in the declarative config.

## Using Skills in Declarative Agents

### Register skills in the YAML

```yaml
skills:
  - id: weather
    version: "1.0.0"
    skill_card_path: ../skills/weather/skill-card.json  # Relative to YAML file
```

Relative `skill_card_path` values are resolved against the YAML file's directory (i.e., `base_path`). When using `yaml_content` instead of `yaml_path`, set `base_path` explicitly on the `DeclarativeAgent`. Absolute paths work regardless. See [Path Resolution](declarative-agents.md#path-resolution).

### Discovery node

Use `list_skills` and `load_skill_md` as tools on a `call_llm` node:

```yaml
nodes:
  - name: discover
    type: call_llm
    args:
      llm: { id: my-llm, version: "1.0.0" }
      prompt: 'prompts["discover"]["instructions"]'
      tools:
        - id: list_skills
        - id: load_skill_md
```

### Execution node with loaded skill tools

After discovery, use `use_tools_from_loaded_skills` to bind whatever tools were loaded:

```yaml
nodes:
  - name: execute
    type: call_llm
    args:
      llm: { id: my-llm, version: "1.0.0" }
      prompt: 'prompts["execute"]["instructions"]'
      use_tools_from_loaded_skills: true
```

sherma tracks which tools were loaded by which skills in an internal state key (`__sherma__`). When `use_tools_from_loaded_skills` is true, only tools associated with loaded skills are bound to the LLM.

## Local vs Remote Skills

Skills can be local (files on disk) or remote (served over HTTP):

**Local** -- Set `base_uri` to a relative or absolute path. Files are read from the filesystem.

```json
{
    "base_uri": "./skills/weather",
    "files": ["SKILL.md", "references/api.md"]
}
```

**Remote** -- Set `base_uri` to a URL. Files are fetched via HTTP GET from `base_uri + "/" + file_path`.

```json
{
    "base_uri": "https://skills.example.com/weather",
    "files": ["SKILL.md", "references/api.md"]
}
```

## Programmatic Skill Loading (`load_skills` node)

For agents that need skills loaded before the planning node runs, use the `load_skills` node type. This is an alternative to progressive disclosure where a dedicated node selects and loads skills upfront.

```yaml
nodes:
  - name: find_relevant_skills
    type: call_llm
    args:
      prompt:
        - role: system
          content: 'prompts["find-relevant-skills"]["instructions"]'
        - role: messages
          content: 'state.messages'
      response_format:
        name: SelectedSkills
        description: Skills relevant to the request
        schema:
          type: object
          properties:
            skills:
              type: array
              items:
                type: object
                properties:
                  id: { type: string }
                  version: { type: string }
                required: [id]
          required: [skills]

  - name: load_selected_skills
    type: load_skills
    args:
      skill_ids: 'json(state.messages[size(state.messages) - 1].content)["skills"]'

  - name: plan
    type: call_llm
    args:
      prompt:
        - role: system
          content: 'prompts["plan"]["instructions"]'
        - role: messages
          content: 'state.messages'
      use_tools_from_loaded_skills: true
```

The `load_skills` node:
- Evaluates the `skill_ids` CEL expression to get a list of `{id, version}` objects
- Loads each skill's SKILL.md and registers its tools
- Synthesizes `AIMessage(tool_calls)` + `ToolMessage` pairs into `state.messages`
- Tracks loaded tool IDs in `__sherma__.loaded_tools_from_skills`

Both patterns (progressive disclosure and programmatic loading) can coexist in the same agent.

## Programmatic Usage

You can also use skill tools outside of declarative agents:

```python
from sherma import create_skill_tools
from sherma.registry.skill import SkillRegistry
from sherma.registry.tool import ToolRegistry

skill_registry = SkillRegistry()
tool_registry = ToolRegistry()

# Register skills (with skill_card attribute)...
# Then create tools:
tools = create_skill_tools(
    skill_registry=skill_registry,
    tool_registry=tool_registry,
    hook_manager=hook_manager,  # Optional
)
```

The returned tools can be bound to any LangChain/LangGraph model.
