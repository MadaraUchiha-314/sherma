<p align="center">
  <img src="https://cdn.wikimg.net/en/hkwiki/images/7/71/Sherma.png" alt="sherma logo" width="200" />
</p>

# sherma

Agent Framework rising through the citadel!

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Language |
| uv | Package manager |
| hatchling | Build backend |
| ruff | Linter and formatter |
| pyright | Type checker |
| pytest | Testing framework |
| commitizen | Conventional commits and version bumps |
| pre-commit | Git hook management |
| GitHub Actions | CI/CD |

## Installation

```bash
pip install sherma
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add sherma
```

Requires Python 3.13+.

## Local Development

```bash
# Clone the repo
git clone https://github.com/MadaraUchiha-314/sherma.git
cd sherma

# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Running Checks

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format --check .

# Type check
uv run pyright

# Unit tests
uv run pytest

# All pre-commit hooks
uv run pre-commit run --all-files
```

## Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/). Commit messages must follow the format:

```
<type>(optional scope): <description>
```

Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

Commitizen enforces this via a commit-msg hook and handles automatic version bumping on merge to main.
