"""Integration tests for message conversion round-trips."""

import pytest
from a2a.types import (
    DataPart,
    FilePart,
    FileWithUri,
    Message,
    Part,
    Role,
    TextPart,
)

from sherma.messages.converter import a2a_to_langgraph, langgraph_to_a2a


@pytest.mark.integration
def test_full_round_trip_with_metadata():
    original = Message(
        message_id="msg-42",
        task_id="task-7",
        context_id="ctx-3",
        role=Role.user,
        parts=[
            Part(root=TextPart(text="Analyze this data")),
            Part(root=DataPart(data={"values": [1, 2, 3], "type": "list"})),
            Part(
                root=FilePart(
                    file=FileWithUri(
                        name="report.csv",
                        mime_type="text/csv",
                        uri="https://example.com/report.csv",
                    ),
                ),
            ),
        ],
    )

    lg_messages = a2a_to_langgraph(original)
    assert len(lg_messages) == 1

    restored = langgraph_to_a2a(lg_messages[0])

    assert restored.role == original.role
    assert restored.message_id == original.message_id
    assert restored.task_id == original.task_id
    assert restored.context_id == original.context_id
    assert len(restored.parts) == len(original.parts)

    # First part is text — preserved exactly
    assert restored.parts[0].root.kind == "text"
    assert restored.parts[0].root.text == "Analyze this data"


@pytest.mark.integration
def test_agent_role_round_trip():
    original = Message(
        message_id="resp-1",
        role=Role.agent,
        parts=[Part(root=TextPart(text="Here is your answer."))],
    )

    lg_messages = a2a_to_langgraph(original)
    restored = langgraph_to_a2a(lg_messages[0])

    assert restored.role == original.role
    assert restored.message_id == original.message_id
    assert restored.parts[0].root.kind == "text"
    assert restored.parts[0].root.text == "Here is your answer."
