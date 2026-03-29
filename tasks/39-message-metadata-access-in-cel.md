# Task 39: Message Metadata Access in CEL (`additional_kwargs`)

**Issue:** [#39](https://github.com/MadaraUchiha-314/sherma/issues/39)
**Priority:** P0

## Description

LangChain messages carry metadata in `additional_kwargs` — a dict of arbitrary key-value pairs. Agents use this to tag messages with types (e.g., `"type": "loaded_skills"`, `"type": "approval_decision"`) and attach structured metadata (e.g., `"context"`, `"rationale"`). Conditional routing and node logic often need to inspect these fields.

## Acceptance Criteria

- When converting LangChain message objects to CEL values, include `additional_kwargs` as a nested map
- Also include `type` (the message class: "human", "ai", "system", "tool") so CEL can distinguish message types
- `state.messages[i]["additional_kwargs"]["type"]` evaluates correctly in CEL
- `state.messages[i]["type"]` returns `"ai"`, `"human"`, etc.
- Backward compatible — existing CEL expressions that access `.content` continue to work

## Analysis

The existing `_object_to_dict()` function in `cel_engine.py` already handles LangChain messages via `model_dump()`, which includes both `additional_kwargs` and `type` fields. The feature already works but lacks:

1. Explicit test coverage for `additional_kwargs` access
2. Documentation in docs and skill references

## Changes Required

1. Add tests in `tests/langgraph/declarative/test_cel_engine.py`
2. Update `docs/declarative-agents.md` with message metadata examples
3. Update `skills/sherma/references/declarative-agents.md` (mirror of docs)
4. Update `skills/sherma/SKILL.md` CEL cheat sheet
