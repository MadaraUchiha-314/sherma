"""Unit tests for sherma.schema module."""

import uuid

import pytest
from a2a.types import AgentExtension, DataPart, Message, Part, Role, TextPart
from pydantic import BaseModel

from sherma.exceptions import SchemaValidationError
from sherma.schema import (
    SCHEMA_INPUT_URI,
    SCHEMA_OUTPUT_URI,
    create_agent_output_as_message_part,
    get_agent_input_from_message_part,
    make_schema_data_part,
    schema_to_extension,
    validate_data,
)


class SampleInput(BaseModel):
    name: str
    value: int


class SampleOutput(BaseModel):
    result: str


def test_schema_to_extension_input():
    ext = schema_to_extension(SCHEMA_INPUT_URI, SampleInput)
    assert isinstance(ext, AgentExtension)
    assert ext.uri == SCHEMA_INPUT_URI
    assert ext.params is not None
    assert ext.params["type"] == "object"
    assert "name" in ext.params["properties"]
    assert "value" in ext.params["properties"]


def test_schema_to_extension_output():
    ext = schema_to_extension(SCHEMA_OUTPUT_URI, SampleOutput)
    assert ext.uri == SCHEMA_OUTPUT_URI
    assert ext.params is not None
    assert "result" in ext.params["properties"]


def test_make_schema_data_part():
    part = make_schema_data_part({"name": "test", "value": 42}, SCHEMA_INPUT_URI)
    root = part.root
    assert root.kind == "data"
    assert root.data == {"name": "test", "value": 42}
    assert root.metadata == {"schema_uri": SCHEMA_INPUT_URI}


def test_validate_data_success():
    result = validate_data({"name": "test", "value": 42}, SampleInput)
    assert isinstance(result, SampleInput)
    assert result.name == "test"
    assert result.value == 42


def test_validate_data_failure():
    with pytest.raises(SchemaValidationError):
        validate_data({"name": "test", "value": "not_an_int"}, SampleInput)


def test_validate_data_missing_field():
    with pytest.raises(SchemaValidationError):
        validate_data({"name": "test"}, SampleInput)


def _make_message(*parts: Part, role: Role = Role.user) -> Message:
    return Message(
        message_id=str(uuid.uuid4()),
        role=role,
        parts=list(parts),
    )


class TestExtractData:
    def test_get_agent_input_from_message_part_success(self):
        msg = _make_message(Part(root=DataPart(data={"name": "alice", "value": 7})))
        result = get_agent_input_from_message_part(msg, SampleInput)
        assert isinstance(result, SampleInput)
        assert result.name == "alice"
        assert result.value == 7

    def test_get_agent_input_from_message_part_no_data_part(self):
        msg = _make_message(Part(root=TextPart(text="hello")))
        with pytest.raises(SchemaValidationError, match="No DataPart found"):
            get_agent_input_from_message_part(msg, SampleInput)

    def test_get_agent_input_from_message_part_invalid_data(self):
        msg = _make_message(Part(root=DataPart(data={"name": "alice", "value": "bad"})))
        with pytest.raises(SchemaValidationError):
            get_agent_input_from_message_part(msg, SampleInput)

    def test_get_agent_input_from_message_part_multiple_parts(self):
        msg = _make_message(
            Part(root=TextPart(text="ignore me")),
            Part(root=DataPart(data={"name": "first", "value": 1})),
            Part(root=DataPart(data={"name": "second", "value": 2})),
        )
        result = get_agent_input_from_message_part(msg, SampleInput)
        assert result.name == "first"
        assert result.value == 1


class TestMakeDataMessage:
    def test_create_agent_output_as_message_part_from_model(self):
        model = SampleOutput(result="done")
        msg = create_agent_output_as_message_part(model, SCHEMA_OUTPUT_URI)
        assert msg.role == Role.agent
        assert len(msg.parts) == 1
        root = msg.parts[0].root
        assert root.kind == "data"
        assert root.data == {"result": "done"}
        assert root.metadata == {"schema_uri": SCHEMA_OUTPUT_URI}

    def test_create_agent_output_as_message_part_from_dict(self):
        msg = create_agent_output_as_message_part({"result": "ok"}, SCHEMA_OUTPUT_URI)
        root = msg.parts[0].root
        assert root.kind == "data"
        assert root.data == {"result": "ok"}

    def test_create_agent_output_as_message_part_defaults(self):
        msg = create_agent_output_as_message_part({"x": 1}, SCHEMA_INPUT_URI)
        assert msg.role == Role.agent
        uuid.UUID(msg.message_id)  # valid uuid
        assert msg.task_id is None
        assert msg.context_id is None

    def test_create_agent_output_as_message_part_custom_fields(self):
        msg = create_agent_output_as_message_part(
            {"x": 1},
            SCHEMA_INPUT_URI,
            role=Role.user,
            message_id="msg-123",
            task_id="task-456",
            context_id="ctx-789",
        )
        assert msg.role == Role.user
        assert msg.message_id == "msg-123"
        assert msg.task_id == "task-456"
        assert msg.context_id == "ctx-789"
