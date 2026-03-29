# Plan 39: Message Metadata Access in CEL (`additional_kwargs`)

## Analysis

The existing `_object_to_dict()` → `model_dump()` pipeline in `cel_engine.py` already converts LangChain messages to CEL maps including all public fields (`content`, `type`, `additional_kwargs`, `tool_calls`, etc.). No code changes to the CEL engine are needed.

## Steps

### 1. Add tests for `additional_kwargs` and `type` access

File: `tests/langgraph/declarative/test_cel_engine.py`

- Test accessing `additional_kwargs` on a message with custom metadata
- Test nested access: `state.messages[i]["additional_kwargs"]["type"]`
- Test `type` field returns correct value for AIMessage, HumanMessage, SystemMessage
- Test backward compatibility (existing `.content` access still works — already covered)

### 2. Update documentation

Files:
- `docs/declarative-agents.md` — Add message metadata section with examples
- `skills/sherma/references/declarative-agents.md` — Mirror the docs update
- `skills/sherma/SKILL.md` — Add to CEL cheat sheet if applicable

### 3. Run tests and lint

```bash
uv run pytest tests/langgraph/declarative/test_cel_engine.py -v
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

### 4. Commit and push
