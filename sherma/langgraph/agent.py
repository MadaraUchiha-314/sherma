from __future__ import annotations

import uuid
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from a2a.client.client import ClientEvent
from a2a.client.middleware import ClientCallContext
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskIdParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Interrupt

from sherma.entities.agent.base import Agent
from sherma.logging import get_logger
from sherma.messages.converter import a2a_to_langgraph, langgraph_to_a2a

logger = get_logger(__name__)


class LangGraphAgent(Agent):
    """An agent backed by a LangGraph compiled state graph.

    Subclass and implement ``get_graph`` to return your compiled graph.
    ``send_message`` and ``cancel_task`` are auto-implemented.
    """

    @abstractmethod
    async def get_graph(self) -> CompiledStateGraph:
        """Return the compiled LangGraph state graph."""
        ...

    async def send_message(
        self,
        request: Message,
        *,
        context: ClientCallContext | None = None,
        request_metadata: dict[str, Any] | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncIterator[ClientEvent | Message]:
        """Convert A2A message to LangGraph format, invoke graph, convert back."""
        graph = await self.get_graph()
        lg_messages = a2a_to_langgraph(request)

        result = await graph.ainvoke({"messages": lg_messages})

        interrupts: tuple[Interrupt, ...] | None = result.get("__interrupt__")
        if interrupts:
            parts = [Part(root=TextPart(text=str(i.value))) for i in interrupts]
            interrupt_msg = Message(
                message_id=str(uuid.uuid4()),
                role=Role.agent,
                parts=parts,
            )
            status = TaskStatus(state=TaskState.input_required, message=interrupt_msg)
            task_id = request.task_id or ""
            context_id = request.context_id or ""
            yield (
                Task(id=task_id, context_id=context_id, status=status),
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=status,
                    final=False,
                ),
            )
        else:
            response_messages = result.get("messages", [])
            if response_messages:
                last_message = response_messages[-1]
                yield langgraph_to_a2a(last_message)

    async def cancel_task(
        self,
        request: TaskIdParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task:
        """Request cancellation of the current task."""
        logger.info("Cancel requested for task %s", request.id)
        return None  # type: ignore[return-value]
