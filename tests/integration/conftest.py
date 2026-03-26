"""Shared fixtures and FakeChatModel for integration tests."""

from __future__ import annotations

from typing import Any

from a2a.types import Message, Part, Role, TextPart
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    """A deterministic chat model that returns scripted responses in order.

    Usage::

        model = FakeChatModel(responses=[
            AIMessage(content="Hello!"),
            AIMessage(content="", additional_kwargs={"tool_calls": [...]}),
        ])

    Each call to ``_generate`` (or ``_agenerate``) pops the next response.
    ``bind_tools`` returns ``self`` unchanged — the test author controls
    what comes back.
    """

    responses: list[BaseMessage]
    call_count: int = 0
    received_messages: list[list[BaseMessage]] = []  # noqa: RUF012

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received_messages.append(list(messages))
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChatModel:
        """Return self — tool binding doesn't change scripted responses."""
        return self


class FailingChatModel(BaseChatModel):
    """A chat model that fails a set number of times before succeeding.

    Usage::

        model = FailingChatModel(
            fail_count=2,
            error=RuntimeError("rate limit"),
            success_response=AIMessage(content="ok"),
        )

    The first ``fail_count`` calls raise ``error``, then subsequent calls
    return ``success_response``.
    """

    fail_count: int = 1
    error: Exception = RuntimeError("transient failure")
    success_response: BaseMessage | None = None
    call_count: int = 0

    model_config = {"arbitrary_types_allowed": True}  # noqa: RUF012

    @property
    def _llm_type(self) -> str:
        return "failing-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise self.error
        response = self.success_response or BaseMessage(content="recovered", type="ai")
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> FailingChatModel:
        """Return self — tool binding doesn't change scripted responses."""
        return self


def make_a2a_message(
    text: str,
    message_id: str = "user-1",
    task_id: str | None = None,
    context_id: str | None = None,
) -> Message:
    """Create an A2A user Message with a single TextPart."""
    return Message(
        message_id=message_id,
        parts=[Part(root=TextPart(text=text))],
        role=Role.user,
        task_id=task_id,
        context_id=context_id,
    )


async def collect_events(agent: Any, message: Message) -> list[Any]:
    """Iterate agent.send_message and collect all events."""
    events: list[Any] = []
    async for event in agent.send_message(message):
        events.append(event)
    return events
