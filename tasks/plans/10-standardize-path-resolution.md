# Plan: Standardize Path Resolution with `base_path`

## Context

When using `yaml_content` (instead of `yaml_path`), relative paths like `skill_card_path` resolve from CWD, which is fragile and surprising. Even with `yaml_path`, internal paths (skill card paths, sub-agent YAML paths) resolve from CWD rather than relative to the YAML file. This makes the framework unusable from any working directory other than the project root.

**Goal:** One clear rule -- "all file paths in the YAML are relative to `base_path`" -- where `base_path` is always explicit or derived from the YAML file location.

## Approach

Keep `load_declarative_config()` unchanged (it just parses YAML, doesn't resolve paths). Add `base_path` to `DeclarativeAgent` and `populate_registries()` where actual file resolution happens.

### 1. Add `base_path` field to `DeclarativeAgent` (`agent.py:118-123`)

```python
base_path: Path | None = None
```

In `get_graph()` (~line 127), derive `base_path` before calling `populate_registries`:

```python
base_path = self.base_path
if base_path is None and self.yaml_path is not None:
    base_path = Path(self.yaml_path).resolve().parent
```

Pass it through:
```python
await populate_registries(
    config, self._registries, self.http_async_client, self.hook_manager,
    base_path=base_path,
)
```

### 2. Add `base_path` parameter to `populate_registries()` (`loader.py:239`)

```python
async def populate_registries(
    config: DeclarativeConfig,
    registries: RegistryBundle,
    http_async_client: Any | None = None,
    hook_manager: HookManager | None = None,
    base_path: Path | None = None,
) -> None:
```

### 3. Resolve `skill_card_path` against `base_path` (`loader.py:306-309`)

```python
path = Path(skill_def.skill_card_path)
if not path.is_absolute():
    if base_path is None:
        raise DeclarativeConfigError(
            f"Relative skill_card_path '{skill_def.skill_card_path}' requires "
            f"a base_path. Provide base_path or use an absolute path."
        )
    path = (base_path / path).resolve()
```

Skill card `base_uri` resolution (lines 312-317) stays unchanged -- it already resolves relative to the skill card file's parent, which is correct.

### 4. Resolve sub-agent `yaml_path` against `base_path` (`loader.py:453-460`)

```python
yaml_path = Path(sub_agent_def.yaml_path)
if not yaml_path.is_absolute():
    if base_path is None:
        raise DeclarativeConfigError(
            f"Relative sub-agent yaml_path '{sub_agent_def.yaml_path}' requires "
            f"a base_path. Provide base_path or use an absolute path."
        )
    yaml_path = (base_path / yaml_path).resolve()
```

The sub-agent `DeclarativeAgent` receives an absolute `yaml_path`, so it auto-derives its own `base_path` from that. Nested sub-agents chain correctly.

### 5. Update example YAML files

- `examples/declarative_skill_agent/agent.yaml:47`: `examples/skills/weather/skill-card.json` -> `../skills/weather/skill-card.json`
- `examples/multi_agent/supervisor_agent.yaml:27`: `examples/multi_agent/weather_agent.yaml` -> `weather_agent.yaml`

### 6. Add tests

In `tests/langgraph/declarative/test_loader.py`:
- Test `base_path` resolves relative `skill_card_path` correctly
- Test `base_path` resolves relative sub-agent `yaml_path` correctly
- Test missing `base_path` with relative path raises `DeclarativeConfigError`
- Test absolute paths work regardless of `base_path`

## What stays unchanged

- `load_declarative_config()` -- just parses YAML, no path resolution
- `import_path` (tools, agents, hooks) -- Python `importlib`, uses `sys.path`
- Skill card `base_uri` -- already resolves relative to skill card file's parent
- `SkillResolver` -- uses already-resolved `base_uri`

## Files to modify

1. `sherma/langgraph/declarative/agent.py` -- add `base_path` field, derive in `get_graph()`
2. `sherma/langgraph/declarative/loader.py` -- add `base_path` param to `populate_registries()`, resolve paths
3. `examples/declarative_skill_agent/agent.yaml` -- fix relative path
4. `examples/multi_agent/supervisor_agent.yaml` -- fix relative path
5. `tests/langgraph/declarative/test_loader.py` -- add new test cases

## Verification

```bash
uv run pytest tests/langgraph/declarative/test_loader.py -v
uv run pyright
uv run ruff check .
```
