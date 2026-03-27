# Plan: Programmatic Skill Loading (`load_skills` node type)

**Issue:** #42
**Priority:** P0

## Overview

Add a `load_skills` node type that enables programmatic skill loading. It evaluates a CEL expression to get a list of skill IDs, then for each skill: loads SKILL.md, registers tools, and synthesizes `AIMessage(tool_calls)` + `ToolMessage(result)` into `state.messages` — making it indistinguishable from progressive disclosure.

## Design Decisions

1. **No state pollution** — skill content goes into `state.messages`, not custom fields.
2. **Synthesize tool call round-trip** — one `AIMessage` with N `tool_calls` + N `ToolMessage` responses (batched).
3. **Reuse existing tracking** — `__sherma__.loaded_tools_from_skills` updated via existing mechanism.
4. **Single arg** — `skill_ids`: CEL expression → list of `{id, version}` objects.
5. **Shared helper** — Extract `load_and_register_skill()` from `skill_tools.py` so both progressive disclosure and `load_skills` node share logic.

## Implementation Steps

### Step 1: Extract shared helper in `skill_tools.py`

Extract core logic from `load_skill_md` into `load_and_register_skill()` returning `(content, tool_ids)`. Refactor `load_skill_md` to call this helper.

### Step 2: Schema changes in `schema.py`

- Add `LoadSkillsArgs(BaseModel)` with `skill_ids: str` field.
- Add `"load_skills"` to `NodeDef.type` Literal.
- Add `LoadSkillsArgs` to `NodeDef.args` union.
- Add `"load_skills": LoadSkillsArgs` to `_resolve_args_type` type_map.

### Step 3: Node builder in `nodes.py`

Implement `build_load_skills_node()`:
- Evaluate `args.skill_ids` CEL expression → list of `{id, version}` dicts.
- For each skill, call `load_and_register_skill()`.
- Build one `AIMessage` with all `tool_calls` + individual `ToolMessage` per skill.
- Update `__sherma__.loaded_tools_from_skills`.
- Fire hooks (`node_enter`, `node_exit`).

### Step 4: Agent dispatcher in `agent.py`

Add `"load_skills"` branch in `_build_node()`.

### Step 5: Tests

- Schema parsing test for `load_skills` node.
- `build_load_skills_node` tests: basic, multiple skills, empty list, preserves existing tool IDs.
- Helper `load_and_register_skill` test.

### Step 6: Documentation and skill updates

- `docs/declarative-agents.md` — add `load_skills` node type.
- `docs/skills.md` — add "Programmatic Skill Loading" section.
- `skills/sherma/references/` — mirror doc changes.
- `skills/sherma/SKILL.md` — update if node types are listed.

## Key Considerations

- Default `version` to `"*"` when missing from CEL result.
- `inject_tool_nodes` transform only affects `call_llm` nodes — no conflict.
- If a skill fails to load, log warning and continue with remaining skills.
- Use `uuid4().hex[:8]` for synthetic tool call IDs.
