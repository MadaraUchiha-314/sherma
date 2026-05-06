"""Unit tests for sherma.schema module."""

import uuid

import pytest
from a2a.types import AgentExtension, DataPart, Message, Part, Role, TextPart
from pydantic import BaseModel

from sherma.exceptions import SchemaValidationError
from sherma.schema import (
    SCHEMA_INPUT_URI,
    SCHEMA_OUTPUT_URI,
    create_agent_input_as_message_part,
    create_agent_output_as_message_part,
    get_agent_input_from_message_part,
    get_agent_output_from_message_part,
    make_schema_data_part,
    schema_to_extension,
    validate_against_schema,
    validate_data,
    validate_json_schema_data,
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


def test_make_schema_data_part_with_extra_metadata():
    part = make_schema_data_part(
        {"name": "test", "value": 42},
        SCHEMA_INPUT_URI,
        extra_metadata={"agent_input": True},
    )
    root = part.root
    assert root.kind == "data"
    assert root.metadata == {"schema_uri": SCHEMA_INPUT_URI, "agent_input": True}


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


SAMPLE_JSON_SCHEMA: dict = {
    "title": "EarningsActuals",
    "type": "object",
    "required": ["ticker", "actuals"],
    "additionalProperties": False,
    "properties": {
        "ticker": {"type": "string", "pattern": "^[A-Z.]+$"},
        "actuals": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
}


def test_validate_json_schema_data_success():
    validate_json_schema_data(
        {"ticker": "AAPL", "actuals": {"revenue": 100.0}},
        SAMPLE_JSON_SCHEMA,
    )


def test_validate_json_schema_data_failure_pattern():
    with pytest.raises(SchemaValidationError):
        validate_json_schema_data(
            {"ticker": "lowercase", "actuals": {}},
            SAMPLE_JSON_SCHEMA,
        )


def test_validate_json_schema_data_failure_extra_property():
    with pytest.raises(SchemaValidationError):
        validate_json_schema_data(
            {"ticker": "AAPL", "actuals": {}, "extra": "nope"},
            SAMPLE_JSON_SCHEMA,
        )


def test_validate_json_schema_data_invalid_schema_raises():
    bad_schema = {"type": "not-a-real-type"}
    with pytest.raises(SchemaValidationError, match="Invalid JSON Schema"):
        validate_json_schema_data({"any": "data"}, bad_schema)


def test_schema_to_extension_accepts_json_schema_dict():
    ext = schema_to_extension(SCHEMA_OUTPUT_URI, SAMPLE_JSON_SCHEMA)
    assert isinstance(ext, AgentExtension)
    assert ext.uri == SCHEMA_OUTPUT_URI
    assert ext.params == SAMPLE_JSON_SCHEMA
    assert "EarningsActuals" in (ext.description or "")


def test_validate_against_schema_dispatches():
    # Pydantic path
    validate_against_schema({"name": "x", "value": 1}, SampleInput)
    with pytest.raises(SchemaValidationError):
        validate_against_schema({"name": "x"}, SampleInput)

    # JSON Schema path
    validate_against_schema({"ticker": "AAPL", "actuals": {}}, SAMPLE_JSON_SCHEMA)
    with pytest.raises(SchemaValidationError):
        validate_against_schema({"ticker": "bad"}, SAMPLE_JSON_SCHEMA)


def _make_message(*parts: Part, role: Role = Role.user) -> Message:
    return Message(
        message_id=str(uuid.uuid4()),
        role=role,
        parts=list(parts),
    )


def _input_data_part(data: dict) -> Part:
    """Create a DataPart tagged as agent input."""
    return Part(root=DataPart(data=data, metadata={"agent_input": True}))


def _output_data_part(data: dict) -> Part:
    """Create a DataPart tagged as agent output."""
    return Part(root=DataPart(data=data, metadata={"agent_output": True}))


class TestExtractInput:
    def test_get_agent_input_from_message_part_success(self):
        msg = _make_message(_input_data_part({"name": "alice", "value": 7}))
        result = get_agent_input_from_message_part(msg, SampleInput)
        assert isinstance(result, SampleInput)
        assert result.name == "alice"
        assert result.value == 7

    def test_get_agent_input_from_message_part_no_data_part(self):
        msg = _make_message(Part(root=TextPart(text="hello")))
        with pytest.raises(SchemaValidationError, match="No agent-input DataPart"):
            get_agent_input_from_message_part(msg, SampleInput)

    def test_get_agent_input_from_message_part_invalid_data(self):
        msg = _make_message(_input_data_part({"name": "alice", "value": "bad"}))
        with pytest.raises(SchemaValidationError):
            get_agent_input_from_message_part(msg, SampleInput)

    def test_get_agent_input_from_message_part_multiple_parts(self):
        msg = _make_message(
            Part(root=TextPart(text="ignore me")),
            _input_data_part({"name": "first", "value": 1}),
            _input_data_part({"name": "second", "value": 2}),
        )
        result = get_agent_input_from_message_part(msg, SampleInput)
        assert result.name == "first"
        assert result.value == 1

    def test_get_agent_input_skips_unmarked_data_part(self):
        """A DataPart without agent_input metadata is ignored."""
        msg = _make_message(
            Part(root=DataPart(data={"name": "plain", "value": 99})),
            _input_data_part({"name": "marked", "value": 1}),
        )
        result = get_agent_input_from_message_part(msg, SampleInput)
        assert result.name == "marked"

    def test_get_agent_input_raises_when_only_unmarked(self):
        """Only unmarked DataParts → should raise."""
        msg = _make_message(
            Part(root=DataPart(data={"name": "plain", "value": 99})),
        )
        with pytest.raises(SchemaValidationError, match="No agent-input DataPart"):
            get_agent_input_from_message_part(msg, SampleInput)


class TestExtractOutput:
    def test_get_agent_output_from_message_part_success(self):
        msg = _make_message(
            _output_data_part({"result": "done"}),
            role=Role.agent,
        )
        result = get_agent_output_from_message_part(msg, SampleOutput)
        assert isinstance(result, SampleOutput)
        assert result.result == "done"

    def test_get_agent_output_from_message_part_no_data_part(self):
        msg = _make_message(Part(root=TextPart(text="hello")))
        with pytest.raises(SchemaValidationError, match="No agent-output DataPart"):
            get_agent_output_from_message_part(msg, SampleOutput)

    def test_get_agent_output_skips_unmarked_data_part(self):
        msg = _make_message(
            Part(root=DataPart(data={"result": "plain"})),
            _output_data_part({"result": "marked"}),
        )
        result = get_agent_output_from_message_part(msg, SampleOutput)
        assert result.result == "marked"


class TestMakeInputMessage:
    def test_create_agent_input_from_model(self):
        model = SampleInput(name="alice", value=7)
        msg = create_agent_input_as_message_part(model, SCHEMA_INPUT_URI)
        assert msg.role == Role.user
        assert len(msg.parts) == 1
        root = msg.parts[0].root
        assert root.kind == "data"
        assert root.data == {"name": "alice", "value": 7}
        assert root.metadata["schema_uri"] == SCHEMA_INPUT_URI
        assert root.metadata["agent_input"] is True

    def test_create_agent_input_from_dict(self):
        msg = create_agent_input_as_message_part(
            {"name": "bob", "value": 3}, SCHEMA_INPUT_URI
        )
        root = msg.parts[0].root
        assert root.data == {"name": "bob", "value": 3}
        assert root.metadata["agent_input"] is True

    def test_create_agent_input_defaults(self):
        msg = create_agent_input_as_message_part({"x": 1}, SCHEMA_INPUT_URI)
        assert msg.role == Role.user
        uuid.UUID(msg.message_id)  # valid uuid
        assert msg.task_id is None
        assert msg.context_id is None


class TestMakeOutputMessage:
    def test_create_agent_output_as_message_part_from_model(self):
        model = SampleOutput(result="done")
        msg = create_agent_output_as_message_part(model, SCHEMA_OUTPUT_URI)
        assert msg.role == Role.agent
        assert len(msg.parts) == 1
        root = msg.parts[0].root
        assert root.kind == "data"
        assert root.data == {"result": "done"}
        assert root.metadata["schema_uri"] == SCHEMA_OUTPUT_URI
        assert root.metadata["agent_output"] is True

    def test_create_agent_output_as_message_part_from_dict(self):
        msg = create_agent_output_as_message_part({"result": "ok"}, SCHEMA_OUTPUT_URI)
        root = msg.parts[0].root
        assert root.kind == "data"
        assert root.data == {"result": "ok"}
        assert root.metadata["agent_output"] is True

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
