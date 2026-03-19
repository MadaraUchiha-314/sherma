# Plan: Sherma Skill for Coding Agents

## Context

Sherma consumers need a way for their coding agents (Claude Code, Cursor, etc.) to understand sherma's constructs and generate correct agent code. Currently, the documentation exists in `docs/` but isn't packaged in a format that coding agents can consume as a skill. This task creates a skill in two formats: Claude Code (invocable via `/sherma`) and agentskills.io (discoverable by sherma-based agents).

## Files to Create

### 1. Canonical skill content (single source of truth)

**`skills/sherma/SKILL.md`**

Single SKILL.md with **merged frontmatter** from both formats:
- agentskills.io fields: `name`, `description`, `license: MIT`
- Claude Code fields: `user-invocable: true`, `allowed-tools: Read, Grep, Glob, Bash, Write, Edit, Agent, WebFetch, WebSearch`, `argument-hint: [description-of-agent-to-build]`

Body sections:
1. **Role** — "You are a sherma agent builder"
2. **Input** — Parse `$ARGUMENTS`, if empty ask what agent to build
3. **Decision tree** — Questions to ask vs defaults:
   - MUST ASK: What does the agent do? What tools?
   - ASK IF AMBIGUOUS: Programmatic vs declarative? Multi-agent?
   - DEFAULTS: OpenAI LLM, memory checkpointer, `messages` list state
4. **Quick reference: Declarative YAML schema** — All top-level keys, node types (`call_llm`, `tool_node`, `call_agent`, `data_transform`, `set_state`, `interrupt`), edge types
5. **Quick reference: Programmatic agent** — `LangGraphAgent` subclass pattern
6. **Templates** — Minimal declarative, with tools, multi-agent, skill-based, hooks, A2A server, programmatic
7. **CEL cheat sheet** — Common patterns, gotchas (inner quotes, `size()`, map syntax)
8. **Common gotchas** — Auto-injected tool_node, prompt role formats, import_path requirements
9. **Process** — Step-by-step: read args → ask questions (max 2 rounds) → choose approach → generate files → verify
10. **Full API surface** — Condensed list of all `sherma.__init__` exports grouped by category

### 2. agentskills.io Skill Card

**`skills/sherma/skill-card.json`**
```json
{
    "id": "sherma",
    "version": "1.0.0",
    "name": "Sherma Agent Builder",
    "description": "Build LLM-powered agents with sherma...",
    "base_uri": ".",
    "files": ["SKILL.md", "references/*.md", "assets/*.yaml", "assets/*.py"],
    "mcps": {},
    "local_tools": {}
}
```

### 3. Claude Code symlink

**`.claude/skills/sherma`** → `../../skills/sherma` (directory symlink)

The entire `.claude/skills/sherma` directory is a symlink to `skills/sherma/`. This avoids any content duplication — Claude Code reads the same files as agentskills.io consumers. Git stores symlinks natively.

### 4. Reference docs and assets

**`skills/sherma/references/`** — Copies of existing docs:
- `declarative-agents.md` ← `docs/declarative-agents.md`
- `concepts.md` ← `docs/concepts.md`
- `multi-agent.md` ← `docs/multi-agent.md`
- `skills.md` ← `docs/skills.md`
- `hooks.md` ← `docs/hooks.md`
- `a2a-integration.md` ← `docs/a2a-integration.md`
- `api-reference.md` ← `docs/api-reference.md`
- `getting-started.md` ← `docs/getting-started.md`

**`skills/sherma/assets/`** — Working examples copied from `examples/`:
- `declarative-weather-agent.yaml` ← `examples/declarative_weather_agent/agent.yaml`
- `programmatic-weather-agent.py` ← `examples/weather_agent/main.py`
- `multi-agent-supervisor.yaml` ← `examples/multi_agent/supervisor_agent.yaml`
- `skill-agent.yaml` ← `examples/declarative_skill_agent/agent.yaml`

## Implementation Order

1. Create `skills/sherma/` directory structure (references/, assets/)
2. Copy reference docs into `skills/sherma/references/`
3. Copy example assets into `skills/sherma/assets/`
4. Write `skills/sherma/skill-card.json`
5. Write `skills/sherma/SKILL.md` (merged frontmatter — single source of truth)
6. Create directory symlink: `.claude/skills/sherma` → `../../skills/sherma`

## Key Design Decisions

- **Inline documentation**: SKILL.md contains enough for an agent to generate correct code without reading other files. References provide deep dives.
- **Follow-up questions**: The skill instructs the agent to ask 1-2 rounds of clarifying questions for genuinely ambiguous decisions, but make defaults for common choices.
- **No content duplication**: Single SKILL.md with merged frontmatter. Claude Code and agentskills.io each read the fields they recognize and ignore the rest.
- **Directory symlink over file symlink**: `.claude/skills/sherma` symlinks to the entire `skills/sherma/` directory, not just SKILL.md. This ensures Claude Code has access to all skill files (references, assets) through the same path.
- **Reference copies**: Docs are copied (not symlinked) into `skills/sherma/references/` for portability — the agentskills.io skill should be self-contained.

## Plan Revisions

### Original plan (pre-implementation)
- Two separate SKILL.md files: `.claude/skills/sherma/SKILL.md` (Claude Code version) and `skills/sherma/SKILL.md` (agentskills.io version) with ~80% shared content.

### Revision 1: Eliminate duplicate SKILL.md
- User pointed out that having two SKILL.md files is unnecessary duplication.
- Changed to: single SKILL.md in `skills/sherma/` with merged frontmatter from both formats, symlinked from `.claude/skills/sherma/SKILL.md`.

### Revision 2: Symlink the whole directory
- User requested the entire folder be symlinked, not just the file.
- Changed to: `.claude/skills/sherma` → `../../skills/sherma` (directory symlink). This is cleaner and gives Claude Code access to references/assets too.

## Verification

1. Invoke `/sherma build a weather agent` in Claude Code — should ask clarifying questions then generate correct YAML + Python
2. Invoke `/sherma` with no args — should ask what to build
3. Verify `skills/sherma/skill-card.json` `files` list matches actual files on disk
4. Run `uv run pytest` to ensure no regressions
5. Run `uv run ruff check .` and `uv run ruff format --check .` for lint/format
