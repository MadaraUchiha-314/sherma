# Plan: Setup Python Repository

## Context

The `sherma` repo is freshly initialized with only `.gitignore`, `LICENSE`, `README.md`, `CLAUDE.md`, and `tasks/`. No Python infrastructure exists. This plan sets up the full Python development environment, CI/CD, and project structure per `tasks/0-setup-python-repo.md`.

## Files to Create

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, build system (hatchling), tool configs |
| `.python-version` | Pin to `3.13` |
| `sherma/__init__.py` | Package init, `__version__`, re-export `hello` |
| `sherma/hello.py` | Hello world function |
| `sherma/py.typed` | PEP 561 type marker (empty) |
| `tests/__init__.py` | Empty |
| `tests/test_hello.py` | Unit tests for hello function |
| `docs/README.md` | Consumer documentation |
| `.pre-commit-config.yaml` | Hooks for ruff, pyright, pytest, commitizen |
| `.github/workflows/pr.yml` | CI on PRs: lint, typecheck, unit + integration tests |
| `.github/workflows/release.yml` | On merge to main: all checks + version bump + PyPI publish |

## Files to Modify

| File | Change |
|------|--------|
| `README.md` | Replace with tech stack table + local dev instructions |
| `CLAUDE.md` | Append tech stack, common commands, project structure |

## Implementation Steps

### 1. Create `pyproject.toml`
- Build backend: `hatchling`
- `requires-python = ">=3.13"`
- `[dependency-groups] dev`: ruff, pyright, pytest, pre-commit, commitizen
- `[tool.ruff]`: select E, W, F, I, B, UP, RUF rules
- `[tool.pyright]`: standard mode, include `sherma/`
- `[tool.pytest.ini_options]`: testpaths=tests, integration marker
- `[tool.commitizen]`: conventional commits, version tracking in pyproject.toml + `sherma/__init__.py`, tag format `v$version`
- License: MIT, Author: Rohith Ramakrishnan

### 2. Create `.python-version`
- Content: `3.13`

### 3. Create `sherma/` package
- `__init__.py`: version `0.1.0`, re-export `hello` from `sherma.hello`
- `hello.py`: `def hello(name: str = "world") -> str` returns `f"Hello, {name}!"`
- `py.typed`: empty marker file

### 4. Create `tests/`
- `__init__.py`: empty
- `test_hello.py`: test default greeting + custom name greeting

### 5. Create `docs/README.md`
- Installation instructions (`pip install sherma`)
- Quick start example using `hello()`

### 6. Create `.pre-commit-config.yaml`
- `ruff-pre-commit` repo: `ruff` (with --fix) + `ruff-format` hooks
- Local hooks: `pyright` via `uv run pyright`, `pytest` via `uv run pytest -m "not integration"`
- `commitizen` repo: commit-msg stage hook

### 7. Create GitHub Actions workflows

**`.github/workflows/pr.yml`** (on pull_request to main):
- Setup: checkout, install uv (astral-sh/setup-uv@v5), uv python install, uv sync
- Steps: ruff check, ruff format --check, pyright, pytest unit, pytest integration

**`.github/workflows/release.yml`** (on push to main):
- Job 1 (checks): same as PR workflow
- Job 2 (release, needs checks):
  - `fetch-depth: 0` for commitizen history
  - `uv run cz bump --yes --changelog` (skip if no bumpable commits)
  - Push version bump commit + tags to main
  - `uv build` then `pypa/gh-action-pypi-publish@release/v1` (OIDC trusted publisher)
  - Permissions: `contents: write`, `id-token: write`

### 8. Update `README.md`
- Tech stack table (Python 3.13, uv, hatchling, ruff, pyright, pytest, commitizen, pre-commit, GitHub Actions)
- Local dev setup: clone, `uv sync`, `uv run pre-commit install --hook-type pre-commit --hook-type commit-msg`
- Running checks: lint, format, typecheck, tests
- Commit conventions explanation

### 9. Update `CLAUDE.md`
- Append: tech stack summary, common commands, project structure

### 10. Install and verify
```bash
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## Verification

1. `uv run ruff check .` - passes with no errors
2. `uv run ruff format --check .` - all files formatted
3. `uv run pyright` - no type errors
4. `uv run pytest` - both tests pass
5. `uv run pre-commit run --all-files` - all hooks pass

## Notes

- PyPI trusted publisher must be configured manually by the repo owner at pypi.org
- `rev` values in `.pre-commit-config.yaml` will be set to match installed versions after `uv sync`
- If branch protection is enabled on main, the release workflow may need a PAT instead of GITHUB_TOKEN for pushing version bumps
