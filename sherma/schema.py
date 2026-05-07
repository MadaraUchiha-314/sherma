"""Core schema module for agent input/output schema support.

Provides constants, helpers, and validation for structured data exchange
between agents using the A2A protocol's DataPart and AgentExtension mechanisms.
"""

from __future__ import annotations

import uuid
from typing import Any

import jsonschema
from a2a.types import AgentExtension, DataPart, Message, Part, Role
from pydantic import BaseModel, ValidationError

from sherma.exceptions import SchemaValidationError

SCHEMA_INPUT_URI = "urn:sherma:schema:input"
SCHEMA_OUTPUT_URI = "urn:sherma:schema:output"


def schema_to_extension(
    uri: str, schema: type[BaseModel] | dict[str, Any]
) -> AgentExtension:
    """Convert a schema to an A2A AgentExtension with its JSON schema.

    *schema* may be either a Pydantic model class (in which case
    ``model_json_schema()`` is used) or a raw JSON Schema dict.
    """
    if isinstance(schema, dict):
        title = schema.get("title", "<json-schema>")
        return AgentExtension(
            uri=uri,
            description=f"JSON Schema for {title}",
            params=schema,
        )

    return AgentExtension(
        uri=uri,
        description=f"JSON Schema for {schema.__name__}",
        params=schema.model_json_schema(),
    )


def make_schema_data_part(
    data: dict[str, Any],
    schema_uri: str,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> Part:
    """Create a DataPart with schema_uri in metadata."""
    metadata: dict[str, Any] = {"schema_uri": schema_uri}
    if extra_metadata:
        metadata.update(extra_metadata)
    return Part(root=DataPart(data=data, metadata=metadata))


def validate_data[T: BaseModel](data: dict[str, Any], schema_model: type[T]) -> T:
    """Validate a dict against a Pydantic model.

    Returns the validated model instance.
    Raises SchemaValidationError if validation fails.
    """
    try:
        return schema_model.model_validate(data)
    except ValidationError as exc:
        raise SchemaValidationError(str(exc)) from exc


def validate_json_schema_data(data: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate a dict against a raw JSON Schema dict.

    Raises :class:`SchemaValidationError` (with the underlying
    ``jsonschema.ValidationError`` message) if validation fails.
    """
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        raise SchemaValidationError(exc.message) from exc
    except jsonschema.SchemaError as exc:
        raise SchemaValidationError(f"Invalid JSON Schema: {exc.message}") from exc


def validate_against_schema(
    data: dict[str, Any], schema: type[BaseModel] | dict[str, Any]
) -> None:
    """Validate *data* against either a Pydantic model or a JSON Schema dict.

    Discards the validated model instance; use :func:`validate_data` directly
    when you need it.
    """
    if isinstance(schema, dict):
        validate_json_schema_data(data, schema)
    else:
        validate_data(data, schema)


def _find_data_part_with_marker(message: Message, marker: str) -> DataPart | None:
    """Find the first DataPart in a message that has a boolean marker in metadata."""
    for part in message.parts:
        root = part.root
        if (
            isinstance(root, DataPart)
            and root.metadata is not None
            and root.metadata.get(marker) is True
        ):
            return root
    return None


def get_agent_input_from_message_part[T: BaseModel](
    message: Message, schema_model: type[T]
) -> T:
    """Extract and validate the first agent-input DataPart from a message.

    Scans message parts for the first DataPart with ``agent_input: true``
    in its metadata, validates its data against the given schema model,
    and returns a typed instance.

    Raises SchemaValidationError if no matching DataPart is found or
    validation fails.
    """
    data_part = _find_data_part_with_marker(message, "agent_input")
    if data_part is not None:
        return validate_data(data_part.data, schema_model)
    raise SchemaValidationError("No agent-input DataPart found in message")


def get_agent_output_from_message_part[T: BaseModel](
    message: Message, schema_model: type[T]
) -> T:
    """Extract and validate the first agent-output DataPart from a message.

    Scans message parts for the first DataPart with ``agent_output: true``
    in its metadata, validates its data against the given schema model,
    and returns a typed instance.

    Raises SchemaValidationError if no matching DataPart is found or
    validation fails.
    """
    data_part = _find_data_part_with_marker(message, "agent_output")
    if data_part is not None:
        return validate_data(data_part.data, schema_model)
    raise SchemaValidationError("No agent-output DataPart found in message")


def _dump_data(data: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Dump a Pydantic model to a dict, or return the dict as-is."""
    if isinstance(data, BaseModel):
        return data.model_dump()
    return data


def create_agent_input_as_message_part(
    data: BaseModel | dict[str, Any],
    schema_uri: str,
    *,
    role: Role = Role.user,
    message_id: str | None = None,
    task_id: str | None = None,
    context_id: str | None = None,
) -> Message:
    """Create a Message containing a single agent-input DataPart.

    Accepts a Pydantic model (auto-dumped) or a dict. Wraps the data in a
    DataPart with ``agent_input: true`` in metadata and returns a complete
    A2A Message.
    """
    return Message(
        message_id=message_id or str(uuid.uuid4()),
        role=role,
        parts=[
            make_schema_data_part(
                _dump_data(data),
                schema_uri,
                extra_metadata={"agent_input": True},
            )
        ],
        task_id=task_id,
        context_id=context_id,
    )


def create_agent_output_as_message_part(
    data: BaseModel | dict[str, Any],
    schema_uri: str,
    *,
    role: Role = Role.agent,
    message_id: str | None = None,
    task_id: str | None = None,
    context_id: str | None = None,
) -> Message:
    """Create a Message containing a single agent-output DataPart.

    Accepts a Pydantic model (auto-dumped) or a dict. Wraps the data in a
    DataPart with ``agent_output: true`` in metadata and returns a complete
    A2A Message.
    """
    return Message(
        message_id=message_id or str(uuid.uuid4()),
        role=role,
        parts=[
            make_schema_data_part(
                _dump_data(data),
                schema_uri,
                extra_metadata={"agent_output": True},
            )
        ],
        task_id=task_id,
        context_id=context_id,
    )
