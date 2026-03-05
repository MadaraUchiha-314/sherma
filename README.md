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
