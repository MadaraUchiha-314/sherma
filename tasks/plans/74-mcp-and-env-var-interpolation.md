# Plan 74: First-class MCP support and env-var interpolation in YAML

## Steps

### Part A — Env-var interpolation (`${VAR}` / `${VAR:-default}`)

1. **Add `_interpolate_env_vars` helper in
   `sherma/langgraph/declarative/loader.py`.**
   - Operate on the parsed Python data (dict/list/scalar tree) returned
     by `yaml.safe_load`, not on the raw YAML text — this keeps YAML
     syntax intact and only substitutes within string values.
   - Pattern: `\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}` matches
     `${VAR}` and `${VAR:-default}`. Multiple substitutions per string
     are supported.
   - Escape: a literal `$$` becomes a single `$` and is not processed.
   - Missing `${VAR}` (with no default) accumulates the variable name.
     If any are missing, raise `DeclarativeConfigError` listing all
     missing names.
   - Recurse into dicts and lists; leave non-string scalars alone.

2. **Call the helper from `load_declarative_config`** right after
   `yaml.safe_load` and before `_parse_config`.

3. **Tests** — `tests/langgraph/declarative/test_env_interpolation.py`
   (new file):
   - Substitute `${VAR}` from `monkeypatch.setenv`.
   - Default fallback `${VAR:-foo}` when var is unset.
   - Missing required var raises with all missing names listed.
   - Substitution inside nested dicts and lists.
   - `$$` escapes a literal `$`.
   - Strings with no `${...}` pass through untouched.

### Part B — First-class MCP support

1. **Schema (`sherma/langgraph/declarative/schema.py`).**
   Add `MCPServerDef`:
   ```python
   class MCPServerDef(BaseModel):
       id: str
       version: str = "*"
       transport: Literal["streamable_http", "sse", "stdio"] = "streamable_http"
       # HTTP/SSE
       url: str | None = None
       headers: dict[str, str] = Field(default_factory=dict)
       # stdio
       command: str | None = None
       args: list[str] = Field(default_factory=list)
       env: dict[str, str] = Field(default_factory=dict)
       # Optional rename to avoid collisions across servers
       tool_prefix: str | None = None
   ```
   Add a `model_validator` that requires `url` for HTTP transports and
   `command` for stdio.
   Add `mcp_servers: list[MCPServerDef] = Field(default_factory=list)`
   to `DeclarativeConfig`.

2. **Loader (`sherma/langgraph/declarative/loader.py`).**
   - Add `_register_mcp_servers(config, registries, tenant_id)`:
     - Build a `MultiServerMCPClient` connection dict per server using
       `langchain_mcp_adapters.client.MultiServerMCPClient`.
     - For each server, call `await client.get_tools(server_name=...)`
       to fetch its tool list.
     - For each LangChain `BaseTool` returned, optionally rename to
       `<tool_prefix><tool_name>`, then wrap via `from_langgraph_tool`,
       set tenant_id, and add to `registries.tool_registry` with
       `id=<final_name>` and `version=server.version`.
   - Call this helper from `populate_registries` after LLM/prompt
     registration but before tools / sub-agents (so MCP tools are in
     the registry by the time other entities load).
   - Wrap connection failures in `DeclarativeConfigError`.

3. **Tests** — `tests/langgraph/declarative/test_mcp_servers.py`
   (new file). Cover purely loader-level behaviour by
   monkeypatching `MultiServerMCPClient.get_tools` to return a fake
   tool list:
   - `mcp_servers` empty / absent: no-op.
   - HTTP server: tools registered with raw names.
   - `tool_prefix` applied to registered ids.
   - stdio server config: stub-tested for connection-dict shape.
   - Schema validator: HTTP without `url` raises; stdio without
     `command` raises.

4. **Schema enforcement note** — keep the changes additive: existing
   YAMLs without `mcp_servers:` continue to work unchanged.

### Part C — Docs and skill (always required by `CLAUDE.md`)

1. **`docs/declarative-agents.md`** — add two new sections:
   - "Environment-variable interpolation" near the top (under
     "YAML Structure"), with examples of `${VAR}` and
     `${VAR:-default}` and the `$$` escape.
   - "MCP servers" alongside "Tools" / "Skills", showing the YAML
     shape for HTTP and stdio servers and how the resulting tools are
     consumed by `call_llm` (via `tools:` or
     `use_tools_from_registry: true`).
2. **`docs/api-reference.md`** — add the `MCPServerDef` Pydantic shape
   and the new `mcp_servers` field on `DeclarativeConfig`.
3. **Mirror to `skills/sherma/references/declarative-agents.md` and
   `skills/sherma/references/api-reference.md`** (these are docs
   copies per `CLAUDE.md`).
4. **`skills/sherma/SKILL.md`** — add `mcp_servers:` to the top-level
   keys list and a one-line note on `${VAR}` interpolation in the
   "Quick Reference" / gotchas section.

### Part D — Verification

1. `uv run ruff check .`
2. `uv run ruff format --check .`
3. `uv run pyright`
4. `uv run pytest -m "not integration"`

### Part E — Commit & PR

1. Conventional-commit message: `feat: add MCP servers and env-var
   interpolation in declarative YAML`.
2. Push to `claude/fix-issue-74-bqsmJ` (the branch the harness has
   pre-assigned for issue #74).
3. Open PR titled `feat: add MCP servers and env-var interpolation in
   declarative YAML`, body referencing issue #74 and describing both
   features.

## Plan Revisions

_None yet._
