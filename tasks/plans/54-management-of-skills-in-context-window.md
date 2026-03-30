# Plan: Management of Skills in the Context Window (#54)

## Problem

When an agent loads skills via `load_skill_md`, they accumulate in the context window. There's no way to:
1. **Unload** a skill (remove its tools and presence from context)
2. **Preserve** loaded skills independently of message compaction (skills embedded in messages can be lost when context is compacted)

## Approach

### 1. `unload_skill_md` Tool

Add an `unload_skill_md` tool to `create_skill_tools()` that:
- Removes the skill's tool IDs from `__sherma__.loaded_tools_from_skills`
- Removes the skill's tool entries from the `ToolRegistry`
- Tracks unloaded skills in `__sherma__.loaded_skills` metadata
- Returns a confirmation message

### 2. Separate Loaded Skills State Tracking

Currently, loaded skill tool IDs are tracked in `__sherma__.loaded_tools_from_skills`. Extend `__sherma__` to also track loaded skill metadata in `__sherma__.loaded_skills`:

```python
__sherma__ = {
    "loaded_tools_from_skills": ["tool_a", "tool_b"],
    "loaded_skills": {
        "weather": {"version": "==1.0.0", "tools": ["get_weather"]},
    }
}
```

This provides a mapping from skill_id → its tools and metadata, making unloading straightforward.

### 3. Handle `unload_skill_md` in `tool_node`

The `tool_node` currently intercepts `load_skill_md` calls to update internal state. Add similar interception for `unload_skill_md`:
- When `unload_skill_md` is called, remove the skill's tools from `loaded_tools_from_skills`
- Remove the skill entry from `loaded_skills`

### 4. Hook Types for Skill Unload

Add `before_skill_unload` and `after_skill_unload` hook types following the existing pattern.

### 5. Update Declarative Skill Agent Example

Update `examples/declarative_skill_agent/agent.yaml` system prompt to inform the agent it can unload skills.

### 6. Update Docs and Skill References

Update `docs/skills.md` and `skills/sherma/references/skills.md` to document the new `unload_skill_md` tool and skill lifecycle management.

## Files to Change

- `sherma/langgraph/skill_tools.py` — Add `unload_skill_md` tool and `unload_and_deregister_skill` function
- `sherma/langgraph/declarative/nodes.py` — Handle `unload_skill_md` in tool_node, update load tracking
- `sherma/hooks/types.py` — Add `BeforeSkillUnloadContext`, `AfterSkillUnloadContext`, hook enum entries
- `sherma/registry/base.py` — Add `remove()` method to Registry
- `examples/declarative_skill_agent/agent.yaml` — Update system prompt
- `tests/langgraph/test_skill_tools.py` — Add tests for unload
- `docs/skills.md` — Document unload capability
- `skills/sherma/references/skills.md` — Mirror docs update
- `skills/sherma/SKILL.md` — Update if skill tools listing changes
