# Plan: Accessing state in CEL should be through state prefix

## Overview

Nest all state fields under a `state` key in the CEL activation so that CEL expressions must use `state.field` or `state["field"]` to access state. Extra variables (`prompts`, `llms`) remain at the top level.

## Steps

### 1. Update `CelEngine._build_activation()` in `cel_engine.py`

Instead of spreading state fields at the top level of the activation dict, nest them under a `"state"` key:

```python
def _build_activation(self, state: dict[str, Any]) -> dict[str, Any]:
    activation: dict[str, Any] = {}
    activation["state"] = _python_to_cel(state)
    for key, value in self._extra_vars.items():
        activation[key] = _python_to_cel(value)
    return activation
```

### 2. Update `nodes.py` raw state shortcut for messages

In `call_llm_fn`, the shortcut `if item.content in state` checks if the CEL expression is a bare state key. With the prefix, `content` will be `state.messages` or `state["messages"]`, so update the shortcut to extract the key after `state.` or `state["..."]` prefix.

### 3. Update all YAML examples

All CEL expressions referencing state fields need `state.` prefix:
- `examples/declarative_weather_agent/agent.yaml`
- `examples/declarative_skill_agent/agent.yaml`
- `examples/declarative_hooks_agent/agent.yaml`
- `examples/multi_agent/weather_agent.yaml`
- `examples/multi_agent/supervisor_agent.yaml`

### 4. Update all tests

Update CEL expressions in test files:
- `tests/langgraph/declarative/test_cel_engine.py`
- `tests/langgraph/declarative/test_nodes.py`
- `tests/langgraph/declarative/test_edges.py`
- `tests/langgraph/declarative/test_schema.py`
- `tests/langgraph/declarative/test_loader.py`
- `tests/langgraph/declarative/test_transform.py`
- `tests/langgraph/declarative/test_agent.py`

### 5. Update docs and skill references

- `docs/declarative-agents.md`
- `skills/sherma/SKILL.md`
- `skills/sherma/references/declarative-agents.md`

### 6. Run tests and linting

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

### 7. Commit and push
