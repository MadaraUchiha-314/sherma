# Task: Programmatic Skill Loading (`load_skills` node type)

**Issue:** #42
**Priority:** P0

## Description

Add a `load_skills` node type that enables programmatic skill selection and loading. Instead of the progressive-disclosure pattern where the LLM drives skill discovery, this node evaluates a CEL expression to get a list of skill IDs and loads them before the planning node runs.

## Changes

- **`sherma/langgraph/skill_tools.py`**: Extracted `load_and_register_skill()` shared helper from `load_skill_md` tool. Both progressive disclosure and the new node reuse this logic.
- **`sherma/langgraph/declarative/schema.py`**: Added `LoadSkillsArgs` model and `"load_skills"` to `NodeDef.type` literal, args union, and type map.
- **`sherma/langgraph/declarative/nodes.py`**: Implemented `build_load_skills_node()` that evaluates CEL, loads skills, and synthesizes `AIMessage(tool_calls)` + `ToolMessage` pairs into messages.
- **`sherma/langgraph/declarative/agent.py`**: Added `"load_skills"` branch to `_build_node()` dispatcher.
- **Tests**: Schema parsing test, node builder tests (basic, multiple skills, empty list, preserves existing tool IDs, default version, skips failed skills).
- **Docs**: Updated `docs/declarative-agents.md`, `docs/skills.md`, skill references, and `SKILL.md`.

## Chat Iterations

1. Initial discussion on whether a generic `call_tool` node could replace `load_skills` — concluded both are complementary.
2. Revised design to avoid state pollution: skill content goes into `state.messages` as synthesized `AIMessage(tool_calls)` + `ToolMessage` pairs instead of custom state fields.
