from __future__ import annotations

import uuid
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from a2a.client.client import UpdateEvent
from a2a.client.middleware import ClientCallContext
from a2a.types import (
    Message,
    Role,
    Task,
    TaskIdParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt
from pydantic import Field

from sherma.entities.agent.base import Agent
from sherma.hooks.executor import HookExecutor
from sherma.hooks.manager import HookManager
from sherma.hooks.types import AfterGraphInvokeContext, GraphInvokeContext
from sherma.logging import get_logger
from sherma.messages.converter import a2a_to_langgraph, langgraph_to_a2a

logger = get_logger(__name__)


def combine_ai_messages(messages: list[AIMessage | str]) -> AIMessage:
    """Combine multiple AIMessages (or plain strings) into one AIMessage.

    Content from each message is merged into list-form content so
    nothing is lost.  Collapses to a plain string when the result
    contains exactly one text block.
    """
    content_blocks: list[str | dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, str):
            content_blocks.append(msg)
        elif isinstance(msg.content, str):
            content_blocks.append(msg.content)
        elif isinstance(msg.content, list):
            content_blocks.extend(msg.content)

    if len(content_blocks) == 1 and isinstance(content_blocks[0], str):
        return AIMessage(content=content_blocks[0])
    return AIMessage(content=content_blocks)


class LangGraphAgent(Agent):
    """An agent backed by a LangGraph compiled state graph.

    Subclass and implement ``get_graph`` to return your compiled graph.
    ``send_message`` and ``cancel_task`` are auto-implemented.
    """

    hook_manager: HookManager = Field(default_factory=HookManager)
    recursion_limit: int = 25
    max_concurrency: int | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def register_hooks(self, executor: HookExecutor) -> None:
        """Register a hook executor with this agent's hook manager."""
        self.hook_manager.register(executor)

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
    ) -> AsyncIterator[UpdateEvent | Message | Task]:
        """Convert A2A message to LangGraph format, invoke graph, convert back."""
        graph = await self.get_graph()
        lg_messages = a2a_to_langgraph(request)
        logger.info("Invoking graph with %d initial messages", len(lg_messages))

        thread_id = request.context_id or request.task_id or str(uuid.uuid4())
        config: dict[str, Any] = {
            "recursion_limit": self.recursion_limit,
            "configurable": {"thread_id": thread_id},
        }
        if self.max_concurrency is not None:
            config["max_concurrency"] = self.max_concurrency
        if self.tags:
            config["tags"] = self.tags
        if self.metadata:
            config["metadata"] = self.metadata

        graph_input = {"messages": lg_messages}
        if self.hook_manager._executors:
            invoke_ctx = GraphInvokeContext(
                agent_id=self.id,
                thread_id=thread_id,
                config=config,
                input=graph_input,
            )
            invoke_ctx = await self.hook_manager.run_hook(
                "before_graph_invoke", invoke_ctx
            )
            config = invoke_ctx.config

        # Check if graph is in interrupted state
        state_snapshot = await graph.aget_state(config)  # type: ignore[arg-type]
        if state_snapshot.tasks:  # pending interrupts exist
            logger.info(
                "Graph is interrupted, resuming with %d messages",
                len(lg_messages),
            )
            result = await graph.ainvoke(
                Command(resume=lg_messages),
                config=config,  # type: ignore[arg-type]
            )
        else:
            result = await graph.ainvoke(
                graph_input,
                config=config,  # type: ignore[arg-type]
            )

        if self.hook_manager._executors:
            after_ctx = AfterGraphInvokeContext(
                agent_id=self.id,
                thread_id=thread_id,
                config=config,
                input=graph_input,
                result=result,
            )
            after_ctx = await self.hook_manager.run_hook(
                "after_graph_invoke", after_ctx
            )
            result = after_ctx.result

        all_messages = result.get("messages", [])
        logger.info("Graph completed with %d total messages", len(all_messages))
        for i, msg in enumerate(all_messages):
            content = getattr(msg, "content", "")
            msg_type = type(msg).__name__
            logger.debug("  msg[%d] %s: %.150s...", i, msg_type, str(content))

        interrupts: tuple[Interrupt, ...] | None = result.get("__interrupt__")

        if interrupts:
            # Combine interrupt values into a single input_required
            # status update.
            ai_message = combine_ai_messages([intr.value for intr in interrupts])
            a2a_msg = langgraph_to_a2a(ai_message)
            status = TaskStatus(
                state=TaskState.input_required,
                message=Message(
                    message_id=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=a2a_msg.parts,
                ),
            )
            yield TaskStatusUpdateEvent(
                task_id=request.task_id or "",
                context_id=request.context_id or "",
                status=status,
                final=False,
            )
        else:
            response_messages = result.get("messages", [])
            if response_messages:
                yield langgraph_to_a2a(response_messages[-1])

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
