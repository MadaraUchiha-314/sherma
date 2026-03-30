# Plan: Static Skill Discovery (Remove Extra LLM Loop)

GitHub Issue: https://github.com/MadaraUchiha-314/sherma/issues/56

## Problem

In the `declarative_skill_agent` example, the `discover_skills` node (a `call_llm` node) has `list_skills` bound as a tool. The LLM must waste one tool-call round-trip just to call `list_skills` and see available skill metadata — even though this metadata is already known at build time from the skill cards declared in the YAML config.

## Approach

Inject skill catalog metadata as a **CEL extra variable** at build time (alongside `prompts` and `llms`), then use the `template()` CEL function to interpolate it into the `discover_skills` prompt. This eliminates the need for the LLM to call `list_skills` at all.

### Why this approach

- The `_build_cel_extra_vars` method already builds `prompts` and `llms` dicts from config. Adding a `skills` dict follows the same pattern.
- The `template()` CEL function (from task 45) already supports `${key}` substitution in prompt strings.
- No new node types needed — just data flow changes.
- The `list_skills` tool remains available for runtime use in other patterns (e.g., dynamic skill registries), but is removed from the `discover_skills` node in the example.

## Steps

### 1. Add `skills` to CEL extra variables

**File**: `sherma/langgraph/declarative/agent.py` — `_build_cel_extra_vars()`

Add a `skills` dict built from `config.skills`, mapping each skill id to its metadata:

```python
skills: dict[str, dict[str, str]] = {}
for skill_def in config.skills:
    skills[skill_def.id] = {
        "id": skill_def.id,
        "version": skill_def.version,
    }
if skills:
    extra["skills"] = skills
```

However, `SkillDef` only has `id`, `version`, `url`, `skill_card_path` — it does **not** have `name` or `description`. Those come from the skill card JSON file (loaded later in `loader.py`). Two sub-options:

**Option A (Recommended)**: Populate `skills` extra var from the **skill registry** after loading, not from raw config. This means `_build_cel_extra_vars` would need access to the skill registry (or the loader passes the resolved skill metadata). The loader already parses skill cards and creates `SkillFrontMatter` objects with `name` and `description`.

**Option B**: Keep it in `_build_cel_extra_vars` but also read the skill card JSON files there. This duplicates logic from the loader.

**Decision**: Go with **Option A** — extend `_build_cel_extra_vars` to accept the skill registry (or have the loader build the skills extra var and pass it in). The loader at `loader.py:346-395` already resolves skill cards and creates `Skill` entities with `front_matter.name` and `front_matter.description`. After the loader populates the skill registry, build the `skills` CEL extra var from it:

```python
skills_extra: dict[str, dict[str, str]] = {}
for skill_def in config.skills:
    try:
        entry = await registries.skill_registry.get(skill_def.id, skill_def.version)
        skills_extra[skill_def.id] = {
            "id": entry.id,
            "version": entry.version,
            "name": entry.front_matter.name,
            "description": entry.front_matter.description,
        }
    except Exception:
        skills_extra[skill_def.id] = {
            "id": skill_def.id,
            "version": skill_def.version,
        }
if skills_extra:
    extra["skills"] = skills_extra
```

This makes `skills["weather"]["name"]`, `skills["weather"]["description"]`, etc. available in CEL expressions.

### 2. Add a `list_skills` node before `discover_skills` in the example

**File**: `examples/declarative_skill_agent/agent.yaml`

Add a new node that runs before `discover_skills`. This can be a `set_state` or `data_transform` node that injects the skill catalog into the conversation as a system/human message. However, with the CEL extra var approach, we can skip this entirely and just use the template in the prompt.

**Simpler approach**: No new node needed. Just change the `discover_skills` prompt to include the skill catalog via CEL template interpolation:

```yaml
prompts:
  - id: discover-skills
    version: "1.0.0"
    instructions: >
      You have access to a catalog of skills. Here are the available skills:
      ${available_skills}

      Given the user's request:
      1. Call load_skill_md for the most relevant skill from the catalog above.
      2. After loading, respond with a brief text summary.
      When a loaded skill is no longer needed, call unload_skill to
      free context window space and unbind its tools.
      IMPORTANT: Do NOT repeat tool calls. Call each tool at most once,
      then reply with text (no tool calls).
```

And in the node, use `template()` to inject the skill list:

```yaml
- name: discover_skills
  type: call_llm
  args:
    llm:
      id: openai-gpt-4o-mini
      version: "1.0.0"
    prompt:
      - role: system
        content: 'template(prompts["discover-skills"]["instructions"], {"available_skills": string(skills)})'
      - role: messages
        content: 'state.messages'
    tools:
      - id: load_skill_md
      - id: unload_skill
```

Note: `list_skills` is **removed** from the tools list.

### 3. Update the skill asset YAML

**File**: `skills/sherma/assets/skill-agent.yaml`

Apply the same changes as step 2 — this is the copy used as a skill asset/template.

### 4. Update integration tests

**File**: `tests/integration/test_declarative_skill_agent.py`

- Remove expectations for `list_skills` tool call in the `discover_skills` step.
- The mocked LLM response sequence should no longer include a `list_skills` call — the first tool call from `discover_skills` should be `load_skill_md` directly.
- Verify that the system prompt contains the skill catalog metadata.

### 5. Update docs and skill references

**Files**:
- `docs/declarative-agents.md` (if it documents the skill agent pattern)
- `skills/sherma/references/declarative-agents.md`
- `skills/sherma/SKILL.md`

Update any documentation that describes the skill discovery flow to reflect:
- `skills` is now available as a CEL extra variable
- The `discover_skills` prompt can reference skill metadata directly via `skills["id"]["name"]` etc.
- `list_skills` is no longer needed in the `discover_skills` node for the standard pattern

### 6. Keep `list_skills` tool available

The `list_skills` tool itself is **not removed** from the codebase — it's still useful for:
- Runtime discovery of dynamically registered skills
- Other patterns where skill catalog isn't known at build time
- The progressive disclosure pattern where LLM-driven discovery is desired

It's only removed from the `discover_skills` node's tool binding in the example.

## Plan Revisions

_(none yet)_
