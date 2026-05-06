# Plan 75: JSON-Schema-validated agent output in declarative YAML

## Approach

Keep the existing Pydantic-typed `Agent.input_schema` /
`Agent.output_schema` contract and broaden it: each field accepts
either a `type[BaseModel]` (Pydantic) **or** a `dict[str, Any]` (raw
JSON Schema). Validation in `sherma/schema.py` and the A2A executor
dispatches on the value's type. `DeclarativeAgent` populates the
fields from `AgentDef.input_schema` / `AgentDef.output_schema` after
config load.

## Steps

1. **Dependency** — add `jsonschema>=4.0.0` to `dependencies` in
   `pyproject.toml`. It is already transitively required, but making
   it explicit pins it as a contract.

2. **Validation utility (`sherma/schema.py`).**
   - Add `validate_json_schema_data(data: dict, schema: dict) -> None`
     which calls `jsonschema.validate` and re-raises
     `jsonschema.ValidationError` as
     `SchemaValidationError` with a readable message.
   - Update `schema_to_extension(uri, schema)` to accept either a
     Pydantic model class (existing behaviour) or a raw JSON Schema
     dict (new). For dicts, build the extension using the dict
     directly as `params`.

3. **Agent base class (`sherma/entities/agent/base.py`).**
   - Widen `input_schema` and `output_schema` types to
     `type[BaseModel] | dict[str, Any] | None`.
   - `get_card` already calls `schema_to_extension` once each — those
     calls now work for both forms thanks to step 2.

4. **A2A executor (`sherma/a2a/executor.py`).**
   - Replace the inline `validate_data` calls with a small dispatcher
     that uses `validate_data` for Pydantic schemas and
     `validate_json_schema_data` for dict schemas. Keep behaviour
     identical when the schema is `None`.

5. **DeclarativeAgent (`sherma/langgraph/declarative/agent.py`).**
   - After `validate_config`, copy
     `agent_def.input_schema` / `agent_def.output_schema` (dicts, when
     set) onto `self` so the executor sees them on the agent.

6. **Tests.**
   - `tests/test_schema.py`:
     - `validate_json_schema_data` accepts a valid payload.
     - `validate_json_schema_data` raises `SchemaValidationError`
       with details on a bad payload.
     - `schema_to_extension` builds an extension from a JSON Schema
       dict.
   - `tests/entities/test_agent.py` (or new test file):
     - Agent with a dict `output_schema` produces the expected
       extension on `get_card()`.
   - `tests/a2a/test_executor.py` (extend existing):
     - End-to-end-ish test: an agent emits a `DataPart` with
       `agent_output: true`; with a dict `output_schema`, valid data
       passes and invalid data raises.
   - `tests/langgraph/declarative/test_agent.py`:
     - Loading a YAML with `output_schema:` results in the dict being
       attached to the agent instance.

7. **Docs.**
   - `docs/declarative-agents.md` — add an "Agent Input/Output Schema"
     section under "Agent Definition" showing both
     `input_schema:` / `output_schema:` JSON-Schema YAML and how the
     schemas are validated and surfaced via A2A.
   - `docs/api-reference.md` — update the `Agent` and
     `DeclarativeConfig` entries; document the broadened type and the
     new `validate_json_schema_data` helper.
   - Mirror both files to `skills/sherma/references/`.
   - Update `skills/sherma/SKILL.md` quick reference (top-level keys
     comment + a one-paragraph note on schema validation).

8. **Verification.**
   - `uv run ruff check .`
   - `uv run ruff format --check .`
   - `uv run pyright` — only pre-existing streamlit/fastapi errors
     should remain.
   - `uv run pytest -m "not integration"` — full unit suite.

9. **Commit & PR.**
   - Branch: `claude/feat-yaml-output-schema-validation`
     (off `main`, since the user requested this be a follow-up to PR
     #75 rather than a stack).
   - Commit: `feat: validate declarative agent input/output against
     YAML JSON Schema`.
   - PR title (conventional commit): same as commit.
   - PR body references issue #74 and PR #75.

## Plan Revisions

_None yet._
