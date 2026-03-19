# Project: sherma

## Planning Workflow

- Every task has a corresponding plan. Before implementing a task, enter plan mode and create a plan.
- Save all plans to the `tasks/plans/` folder.
- Plans follow the same naming convention as tasks: `<number>-<kebab-case-description>.md` (e.g., `tasks/plans/0-setup-python-repo.md` is the plan for `tasks/0-setup-python-repo.md`).
- After the plan is saved, exit plan mode and implement the plan.

## Documenting Iterations

- All iterations of a task done through chat (feedback, course corrections, design changes) must be documented in the task file under a "Chat Iterations" section.
- All changes to plans (revised decisions, new approaches, dropped ideas) must be documented in the corresponding plan file under a "Plan Revisions" section.

## Docs and Skills

- Documentation (`docs/`) and the skill (`skills/sherma/`) must be updated together. Any task that changes agent behavior, YAML schema, API surface, hooks, or any user-facing feature must include updates to both `docs/` and `skills/sherma/references/` (which are copies of docs) as part of task completion.
- The skill SKILL.md (`skills/sherma/SKILL.md`) must also be updated if the change affects templates, the quick reference, the CEL cheat sheet, gotchas, or the API surface listing.
- Plans must explicitly include a step for updating docs and the skill. If a plan does not have this step, add it before implementation begins.

## Tech Stack

- **Language:** Python 3.13
- **Package manager:** uv
- **Build backend:** hatchling
- **Linter/Formatter:** ruff
- **Type checker:** pyright
- **Testing:** pytest
- **Commits:** commitizen (conventional commits)
- **Git hooks:** pre-commit
- **CI/CD:** GitHub Actions

## Common Commands

```bash
uv sync                              # Install dependencies
uv run ruff check .                  # Lint
uv run ruff format --check .         # Format check
uv run pyright                       # Type check
uv run pytest                        # Run tests
uv run pytest -m "not integration"   # Unit tests only
uv run pytest -m "integration"       # Integration tests only
uv run pre-commit run --all-files    # Run all pre-commit hooks
```

## Project Structure

```
sherma/
  __init__.py       # Package init, __version__, re-exports
  hello.py          # Hello world function
  py.typed          # PEP 561 type marker
tests/
  test_hello.py     # Unit tests
docs/
  README.md         # Consumer documentation
tasks/
  plans/            # Task plans
```
