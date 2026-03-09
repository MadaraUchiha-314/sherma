from sherma.messages.converter import a2a_to_langgraph, langgraph_to_a2a


def test_simple_text_round_trip():
    a2a_msg = {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello"}],
        "messageId": "msg-1",
        "taskId": "task-1",
        "contextId": "ctx-1",
    }
    lg_messages = a2a_to_langgraph(a2a_msg)
    assert len(lg_messages) == 1

    result = langgraph_to_a2a(lg_messages[0])
    assert result["role"] == "user"
    assert result["parts"] == [{"kind": "text", "text": "Hello"}]
    assert result["messageId"] == "msg-1"
    assert result["taskId"] == "task-1"
    assert result["contextId"] == "ctx-1"


def test_agent_role_round_trip():
    a2a_msg = {
        "role": "agent",
        "parts": [{"kind": "text", "text": "I can help."}],
    }
    lg_messages = a2a_to_langgraph(a2a_msg)
    result = langgraph_to_a2a(lg_messages[0])
    assert result["role"] == "agent"
    assert result["parts"] == [{"kind": "text", "text": "I can help."}]


def test_multi_part_round_trip():
    a2a_msg = {
        "role": "user",
        "parts": [
            {"kind": "text", "text": "See this file:"},
            {
                "kind": "file",
                "file": {"name": "doc.pdf", "mimeType": "application/pdf"},
            },
            {"kind": "data", "data": {"key": "value"}},
        ],
    }
    lg_messages = a2a_to_langgraph(a2a_msg)
    result = langgraph_to_a2a(lg_messages[0])
    assert result["role"] == "user"
    assert len(result["parts"]) == 3
    assert result["parts"][0] == {"kind": "text", "text": "See this file:"}
    assert result["parts"][1] == {
        "kind": "file",
        "file": {"name": "doc.pdf", "mimeType": "application/pdf"},
    }
    assert result["parts"][2] == {"kind": "data", "data": {"key": "value"}}


def test_dict_message_conversion():
    lg_msg = {
        "type": "human",
        "content": "Hello from dict",
        "additional_kwargs": {},
    }
    result = langgraph_to_a2a(lg_msg)
    assert result["role"] == "user"
    assert result["parts"] == [{"kind": "text", "text": "Hello from dict"}]


def test_ai_message_conversion():
    lg_msg = {
        "type": "ai",
        "content": "I'm an AI response",
        "additional_kwargs": {},
    }
    result = langgraph_to_a2a(lg_msg)
    assert result["role"] == "agent"
