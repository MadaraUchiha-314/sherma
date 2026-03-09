from a2a.types import Message, Part, Role, TextPart
from langchain_core.messages import AIMessage, HumanMessage

from sherma.messages.converter import a2a_to_langgraph, langgraph_to_a2a


def _make_message(
    text: str,
    role: Role = Role.user,
    message_id: str = "msg-1",
    task_id: str | None = None,
    context_id: str | None = None,
) -> Message:
    return Message(
        message_id=message_id,
        parts=[Part(root=TextPart(text=text))],
        role=role,
        task_id=task_id,
        context_id=context_id,
    )


def test_simple_text_round_trip():
    a2a_msg = _make_message("Hello", task_id="task-1", context_id="ctx-1")
    lg_messages = a2a_to_langgraph(a2a_msg)
    assert len(lg_messages) == 1
    assert isinstance(lg_messages[0], HumanMessage)

    result = langgraph_to_a2a(lg_messages[0])
    assert isinstance(result, Message)
    assert result.role == Role.user
    assert result.parts[0].root.text == "Hello"
    assert result.message_id == "msg-1"
    assert result.task_id == "task-1"
    assert result.context_id == "ctx-1"


def test_agent_role_round_trip():
    a2a_msg = _make_message("I can help.", role=Role.agent)
    lg_messages = a2a_to_langgraph(a2a_msg)
    assert isinstance(lg_messages[0], AIMessage)

    result = langgraph_to_a2a(lg_messages[0])
    assert isinstance(result, Message)
    assert result.role == Role.agent
    assert result.parts[0].root.text == "I can help."


def test_human_message_conversion():
    lg_msg = HumanMessage(content="Hello from langchain")
    result = langgraph_to_a2a(lg_msg)
    assert isinstance(result, Message)
    assert result.role == Role.user
    assert result.parts[0].root.text == "Hello from langchain"


def test_ai_message_conversion():
    lg_msg = AIMessage(content="I'm an AI response")
    result = langgraph_to_a2a(lg_msg)
    assert isinstance(result, Message)
    assert result.role == Role.agent
    assert result.parts[0].root.text == "I'm an AI response"


def test_metadata_preserved():
    lg_msg = HumanMessage(
        content="test",
        additional_kwargs={
            "a2a_metadata": {
                "messageId": "m-42",
                "taskId": "t-1",
                "contextId": "c-1",
            }
        },
    )
    result = langgraph_to_a2a(lg_msg)
    assert result.message_id == "m-42"
    assert result.task_id == "t-1"
    assert result.context_id == "c-1"
