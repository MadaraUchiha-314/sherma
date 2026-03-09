# Plan: Agent Input/Output Schema Support

## Context

Agents sometimes require custom structured input/output beyond plain text messages. We need a way for agents to:
1. Declare input and output JSON schemas
2. Publish these schemas so callers know what to send/expect
3. Send and receive schema-conforming data via `DataPart`

The A2A protocol's `DataPart` (with `data` and `metadata` fields) and `AgentExtension` mechanism provide the building blocks.

## Approach

- Agents declare schemas as Pydantic models (`input_schema`, `output_schema`)
- Schemas are published as A2A extensions on the AgentCard (`urn:sherma:schema:input`, `urn:sherma:schema:output`)
- `DataPart.metadata` carries `schema_uri` to tag data parts with their schema
- Automatic validation at the executor boundary when schemas are declared
- Proper `DataPart` round-trip support in the LangGraph message converter

## Implementation Order

### 1. Add `SchemaValidationError` to `sherma/exceptions.py`

```python
class SchemaValidationError(ShermaError):
    """Raised when data does not conform to a declared schema."""
```

### 2. Create `sherma/schema.py` — Core schema module

Constants and helpers:
- `SCHEMA_INPUT_URI = "urn:sherma:schema:input"`
- `SCHEMA_OUTPUT_URI = "urn:sherma:schema:output"`
- `schema_to_extension(uri, schema_model) -> AgentExtension` — converts Pydantic model to A2A extension via `.model_json_schema()`
- `make_schema_data_part(data, schema_uri) -> Part` — creates a `DataPart` with `metadata={"schema_uri": ...}`
- `validate_data(data, schema_model) -> BaseModel` — validates dict against Pydantic model, raises `SchemaValidationError` on failure

### 3. Modify `sherma/entities/agent/base.py` — Add schema fields

Add to `Agent`:
- `input_schema: type[BaseModel] | None = None`
- `output_schema: type[BaseModel] | None = None`

Update `get_card()` to auto-inject schema extensions into the AgentCard's capabilities when schemas are declared. Must create copies (not mutate) and avoid duplicate extensions.

### 4. Modify `sherma/messages/a2a.py` — Add dict-based helper

Add `make_schema_data_part(data, schema_uri) -> dict` alongside existing `make_data_part`.

### 5. Modify `sherma/messages/converter.py` — DataPart round-trip

Update `_a2a_part_to_content_block`: for `kind == "data"`, produce `{"type": "data", "data": ..., "metadata": ...}` instead of lossy `{"type": "data", "raw": ...}`.

Update `_content_block_to_a2a_part`: for `type == "data"`, reconstruct `DataPart` with data and metadata.

### 6. Modify `sherma/a2a/executor.py` — Automatic validation

In `execute()`, when schemas are present on the agent:
- Before calling agent: if `agent.input_schema` is set, validate incoming `DataPart`s against it
- After receiving response: if `agent.output_schema` is set, validate outgoing `DataPart`s against it

### 7. Modify `sherma/__init__.py` — Re-export new public API

Add: `SchemaValidationError`, `SCHEMA_INPUT_URI`, `SCHEMA_OUTPUT_URI`, `make_schema_data_part`, `validate_data`, `schema_to_extension`

### 8. Tests

- `tests/test_schema.py` — Unit tests for schema module (extension creation, data part creation, validation success/failure)
- `tests/messages/test_converter.py` — Add DataPart round-trip tests
- `tests/entities/agent/test_base.py` — Test schema extension injection in `get_card()`

## Key Files

| File | Action |
|------|--------|
| `sherma/exceptions.py` | MODIFY — add `SchemaValidationError` |
| `sherma/schema.py` | CREATE — schema constants, helpers, validation |
| `sherma/entities/agent/base.py` | MODIFY — add schema fields, update `get_card()` |
| `sherma/messages/a2a.py` | MODIFY — add `make_schema_data_part` dict helper |
| `sherma/messages/converter.py` | MODIFY — DataPart round-trip support |
| `sherma/a2a/executor.py` | MODIFY — automatic schema validation |
| `sherma/__init__.py` | MODIFY — re-export new API |

## Verification

```bash
uv run pytest                        # All tests pass
uv run ruff check .                  # Lint clean
uv run ruff format --check .         # Format clean
uv run pyright                       # Type check clean
uv run pre-commit run --all-files    # All hooks pass
```
