# Plan 30: Skill card should explicitly state its extensions

## Steps

1. **Add `SkillExtension` model to `skill_card.py`** — Modeled after A2A `AgentExtension`
   with `uri`, `description`, `required`, `params`.

2. **Change `SkillCard.extensions`** from `dict[str, Any]` to `list[SkillExtension]`.

3. **Export `SkillExtension`** from `entities/__init__.py` and `sherma/__init__.py`.

4. **Update tests** — Adjust existing tests, add `SkillExtension` unit tests.

5. **Update `examples/skills/weather/skill-card.json`** — Use array-based extensions.

6. **Update docs** — `docs/skills.md`, `docs/api-reference.md`, and their
   `skills/sherma/references/` copies.

7. **Run tests and linting** — Ensure nothing breaks.

## Plan Revisions

1. Original plan used `dict[str, Any]` for extensions. Revised to use
   `list[SkillExtension]` with URI-identified entries per A2A AgentExtension spec.
