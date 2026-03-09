- We need to add support for skills.
- https://agentskills.io/home
    - Read this through thoroughly
- Create LangGraph tools to both list and load skills
- These will be used by LLMs/Agents to discover and progressively load skills
    - https://agentskills.io/specification#progressive-disclosure
- Create the following functions
    - load_skill_md(id, version) -> Markdown
    - load_skill_resource(id, versio, resource_path) -> Markdown
    - load_skill_asset(id, version, asset_path) -> Any
- Provide "list" variants of each of these functions
- Create an entity called skill-card.json
    - This is exactly like agent-card.json

```json
{
    "id": "",
    "version": "",
    "name": "",
    "description": "",
    "base_uri": "",
    "files": [
        "SKILL.md",
        "assets/my-asset-1.png",
        "references/my-reference-1.md",
        "scripts/my-script-1.py"
    ],
    "extensions": [

    ]
}
```

- All the files should be accessible via GET api from the base-uri + the file paths specified in `files`

- Just like tools, agents: skills can also be local or remote.
    - Add support to local and remote skills using the skill_card

- We create an extension to add mcp support
    - Since executing scripts in hosted environments is unsafe, we provide support to just use MCP tools
    - This extension will have 2 attributes additional to extension name etc:
        - url
        - transport
    - Goto through the mcp documentation to define this
    - The idea here is that the "MCP" server gets attched to the agent as Tools when the LLM decides to load the skill

    - Before "planning" an agent may choose to load all the tools related to the loaded "skills" 
    - Provide utilities for this
    - Use MCP Langgraph Adapter for this

```json
"mcps": {
    "my-mcp-server": {
        "id": "",
        "version": "",
        "url": "",
        "transport": ""
    }
}
```

- We want to create an extension for local tools support
- Just like how declarative agent refers to local tools, use the same mechanism

```json
"local_tools": {
    "my-local-tool": {
        "id": "",
        "version": "",
        "import_path": ""
    }
}
```