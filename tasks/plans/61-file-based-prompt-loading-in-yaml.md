# Plan 61: File-based prompt loading in YAML

## Steps

1. **Update `PromptDef` in `sherma/langgraph/declarative/schema.py`**
   - Make `instructions` optional (`str | None = None`).
   - Add `instructions_path: str | None = None`.
   - Add a `model_validator` enforcing exactly one of
     `instructions` or `instructions_path` is set.

2. **Update `populate_registries` in `sherma/langgraph/declarative/loader.py`**
   - When `instructions_path` is set, resolve it against `base_path`
     (relative paths require `base_path`, like `skill_card_path`).
   - Read the file as text and use its contents as the prompt
     instructions. Raise `DeclarativeConfigError` if the file is missing
     or `base_path` is required but not provided.

3. **Tests** — Add unit tests in `tests/langgraph/declarative/test_loader.py`
   and `test_schema.py` covering:
   - Loading a prompt from a relative `instructions_path` with `base_path`.
   - Loading from an absolute `instructions_path`.
   - Error when both `instructions` and `instructions_path` are set.
   - Error when neither is set.
   - Error when relative `instructions_path` is given without `base_path`.
   - Error when the file does not exist.

4. **Docs** — Update `docs/declarative-agents.md` and `docs/api-reference.md`
   with the new field and an example. Mirror changes in
   `skills/sherma/references/`.

5. **Skill SKILL.md** — Update `skills/sherma/SKILL.md` quick reference and
   YAML templates to mention `instructions_path`.

6. **Run `uv run ruff check .`, `uv run ruff format --check .`,
   `uv run pyright`, `uv run pytest -m "not integration"`.**

7. **Commit and push to
   `claude/feat-file-based-prompt-loading-in-yaml-EuSgI`.**

## Plan Revisions

_None yet._
