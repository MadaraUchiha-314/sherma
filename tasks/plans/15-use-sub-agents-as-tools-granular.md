# Plan: Extend `use_sub_agents_as_tools` to support `"all"` and a list of RegistryRefs

## Context

Currently `use_sub_agents_as_tools` is a `bool`. When `true`, ALL declared sub-agents are bound as tools. The request is to support selecting specific sub-agents by ID+version using `RegistryRef` objects. The tool name when wrapping agents as tools is already the agent's `id` (via `agent_to_langgraph_tool`), so the RegistryRef `id` field naturally maps to the tool name.

## YAML usage examples

```yaml
# All sub-agents (backward compatible)
use_sub_agents_as_tools: true

# Explicit "all"
use_sub_agents_as_tools: all

# Specific sub-agents by RegistryRef
use_sub_agents_as_tools:
  - id: weather-agent
    version: "1.0.0"
  - id: search-agent
    version: "1.0.0"
```

## Changes

### 1. `sherma/langgraph/declarative/schema.py` — Type + validator

- Change type from `bool` to `Literal[False, "all"] | list[RegistryRef]` with default `False`
- Add `field_validator` (mode="before") to normalize:
  - `True` (YAML `true`) → `"all"`
  - `False` → `False`
  - `"all"` → `"all"`
  - `list[dict]` → validated as `list[RegistryRef]` by Pydantic
  - anything else → `ValidationError`

### 2. `sherma/langgraph/declarative/nodes.py` — Filter logic (lines 182-188)

- When `"all"`: use all `sub_agent_tool_ids` from context (current behavior)
- When `list[RegistryRef]`: resolve only those specific sub-agent tools from the tool registry using the ref's `id` and `version`

### 3. `sherma/langgraph/declarative/loader.py` — Validation

- Line 647: wrap in `bool()` for the `sum()` mutual exclusivity check
- After line 664: when value is `list[RegistryRef]`, validate each ref's `id` exists in `config.sub_agents`

### 4. No changes needed

- `agent.py` — already passes all sub-agent IDs; filtering is in nodes.py
- `transform.py` — truthiness checks already work

### 5. Tests

- Schema parsing: `true`→`"all"`, `false`→`False`, `"all"`→`"all"`, list of dicts→`list[RegistryRef]`, invalid→error
- Validation: unknown sub-agent ID in list → `DeclarativeConfigError`
- Node behavior: `"all"` resolves all, `list[RegistryRef]` resolves subset, `False` resolves none

## Verification

```bash
uv run pytest -m "not integration"
uv run ruff check .
uv run pyright
```
