# Plan 30: Skill card should explicitly state its extensions

## Steps

1. **Update `examples/skills/weather/skill-card.json`** — Add `extensions` field
   declaring `local_tools` as a used extension.

2. **Update `docs/skills.md`** — Add `extensions` field to the skill card example
   JSON and the field table.

3. **Update `skills/sherma/references/skills.md`** — Mirror the docs change (skill
   reference copy).

4. **Run tests and linting** — Ensure nothing breaks.
