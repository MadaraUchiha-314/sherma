"""Core schema module for agent input/output schema support.

Provides constants, helpers, and validation for structured data exchange
between agents using the A2A protocol's DataPart and AgentExtension mechanisms.
"""

from __future__ import annotations

import uuid
from typing import Any

from a2a.types import AgentExtension, DataPart, Message, Part, Role
from pydantic import BaseModel, ValidationError

from sherma.exceptions import SchemaValidationError

SCHEMA_INPUT_URI = "urn:sherma:schema:input"
SCHEMA_OUTPUT_URI = "urn:sherma:schema:output"


def schema_to_extension(uri: str, schema_model: type[BaseModel]) -> AgentExtension:
    """Convert a Pydantic model to an A2A AgentExtension with its JSON schema."""
    return AgentExtension(
        uri=uri,
        description=f"JSON Schema for {schema_model.__name__}",
        params=schema_model.model_json_schema(),
    )


def make_schema_data_part(data: dict[str, Any], schema_uri: str) -> Part:
    """Create a DataPart with schema_uri in metadata."""
    return Part(root=DataPart(data=data, metadata={"schema_uri": schema_uri}))


def validate_data[T: BaseModel](data: dict[str, Any], schema_model: type[T]) -> T:
    """Validate a dict against a Pydantic model.

    Returns the validated model instance.
    Raises SchemaValidationError if validation fails.
    """
    try:
        return schema_model.model_validate(data)
    except ValidationError as exc:
        raise SchemaValidationError(str(exc)) from exc


def get_agent_input_from_message_part[T: BaseModel](
    message: Message, schema_model: type[T]
) -> T:
    """Extract and validate the first DataPart from a message.

    Scans message parts for the first DataPart, validates its data against
    the given schema model, and returns a typed instance.

    Raises SchemaValidationError if no DataPart is found or validation fails.
    """
    for part in message.parts:
        if part.root.kind == "data":
            return validate_data(part.root.data, schema_model)
    raise SchemaValidationError("No DataPart found in message")


def create_agent_output_as_message_part(
    data: BaseModel | dict[str, Any],
    schema_uri: str,
    *,
    role: Role = Role.agent,
    message_id: str | None = None,
    task_id: str | None = None,
    context_id: str | None = None,
) -> Message:
    """Create a Message containing a single DataPart.

    Accepts a Pydantic model (auto-dumped) or a dict. Wraps the data in a
    DataPart with schema_uri metadata and returns a complete A2A Message.
    """
    if isinstance(data, BaseModel):
        data_dict = data.model_dump()
    else:
        data_dict = data

    return Message(
        message_id=message_id or str(uuid.uuid4()),
        role=role,
        parts=[make_schema_data_part(data_dict, schema_uri)],
        task_id=task_id,
        context_id=context_id,
    )
