from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from sherma.entities.agent.base import Agent
from sherma.logging import get_logger
from sherma.messages.converter import a2a_to_langgraph, langgraph_to_a2a

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

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

    async def send_message(self, message: Any) -> Any:
        """Convert A2A message to LangGraph format, invoke graph, convert back."""
        graph = await self.get_graph()
        lg_messages = a2a_to_langgraph(message)

        result = await graph.ainvoke({"messages": lg_messages})

        response_messages = result.get("messages", [])
        if not response_messages:
            return None

        last_message = response_messages[-1]
        return langgraph_to_a2a(last_message)

    async def cancel_task(self, task_id: str) -> None:
        """Request cancellation of the current task."""
        logger.info("Cancel requested for task %s", task_id)
