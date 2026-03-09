"""Integration tests for message conversion round-trips."""

import pytest

from sherma.messages.converter import a2a_to_langgraph, langgraph_to_a2a


@pytest.mark.integration
def test_full_round_trip_with_metadata():
    original = {
        "role": "user",
        "parts": [
            {"kind": "text", "text": "Analyze this data"},
            {"kind": "data", "data": {"values": [1, 2, 3], "type": "list"}},
            {
                "kind": "file",
                "file": {
                    "name": "report.csv",
                    "mimeType": "text/csv",
                    "uri": "https://example.com/report.csv",
                },
            },
        ],
        "messageId": "msg-42",
        "taskId": "task-7",
        "contextId": "ctx-3",
    }

    lg_messages = a2a_to_langgraph(original)
    assert len(lg_messages) == 1

    restored = langgraph_to_a2a(lg_messages[0])

    assert restored["role"] == original["role"]
    assert restored["messageId"] == original["messageId"]
    assert restored["taskId"] == original["taskId"]
    assert restored["contextId"] == original["contextId"]
    assert len(restored["parts"]) == len(original["parts"])

    assert restored["parts"][0] == original["parts"][0]
    assert restored["parts"][1] == original["parts"][1]
    assert restored["parts"][2] == original["parts"][2]


@pytest.mark.integration
def test_agent_role_round_trip():
    original = {
        "role": "agent",
        "parts": [{"kind": "text", "text": "Here is your answer."}],
        "messageId": "resp-1",
    }

    lg_messages = a2a_to_langgraph(original)
    restored = langgraph_to_a2a(lg_messages[0])

    assert restored == original
