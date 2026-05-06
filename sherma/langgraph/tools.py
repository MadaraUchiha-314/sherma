from __future__ import annotations

from typing import Any

from a2a.types import Message as A2AMessage
from a2a.types import Part, Role, TextPart
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from sherma.entities.agent.base import Agent
from sherma.entities.tool import Tool
from sherma.schema import make_schema_data_part


def from_langgraph_tool(base_tool: BaseTool) -> Tool:
    """Wrap a LangChain BaseTool as a sherma Tool entity."""
    return Tool(
        id=base_tool.name,
        version="*",
        function=base_tool,
    )


def to_langgraph_tool(tool: Tool) -> BaseTool:
    """Wrap a sherma Tool as a LangChain BaseTool.

    If the tool's function is already a BaseTool, return it directly.
    Otherwise, wrap the callable using the @tool decorator pattern.
    """
    if isinstance(tool.function, BaseTool):
        return tool.function

    return StructuredTool.from_function(
        func=tool.function,
        name=tool.id,
        description=f"Sherma tool: {tool.id}",
    )


def agent_to_langgraph_tool(agent: Agent) -> BaseTool:
    """Wrap a sherma Agent as a LangGraph BaseTool.

    If the agent has an ``input_schema``, the tool accepts both a text
    ``request`` and a structured ``agent_input``.  Otherwise it accepts
    only ``request``.

    The tool invokes ``agent.send_message`` with an A2A Message and
    returns the text content of the response.
    """
    description = (
        agent.agent_card.description
        if agent.agent_card and agent.agent_card.description
        else f"Invoke agent: {agent.id}"
    )

    raw_input_schema = agent.input_schema
    # Only Pydantic-model inputs participate in the structured args_schema
    # for the wrapper tool. JSON-Schema-dict inputs are still validated by
    # the A2A executor at the agent boundary, but are not exposed as a
    # structured field on the LangGraph tool's args_schema.
    input_schema: type[BaseModel] | None = (
        raw_input_schema
        if isinstance(raw_input_schema, type) and issubclass(raw_input_schema, BaseModel)
        else None
    )

    if input_schema is not None:
        args_schema: type[BaseModel] = create_model(
            f"{input_schema.__name__}ToolInput",
            request=(
                str,
                Field(description="Natural language request to the agent"),
            ),
            agent_input=(
                input_schema,
                Field(description=f"Structured input: {input_schema.__name__}"),
            ),
        )
    else:
        args_schema = create_model(
            "AgentToolInput",
            request=(
                str,
                Field(description="Natural language request to the agent"),
            ),
        )

    async def _invoke_agent(**kwargs: Any) -> str:
        request_text: str = kwargs["request"]
        agent_input_data: BaseModel | None = kwargs.get("agent_input")

        parts: list[Part] = [Part(root=TextPart(text=request_text))]
        if agent_input_data is not None and input_schema is not None:
            parts.append(
                make_schema_data_part(
                    agent_input_data.model_dump()
                    if isinstance(agent_input_data, BaseModel)
                    else agent_input_data,
                    f"urn:sherma:schema:input:{agent.id}",
                    extra_metadata={"agent_input": True},
                )
            )

        msg = A2AMessage(
            message_id="agent-tool-call",
            parts=parts,
            role=Role.user,
        )

        results: list[Any] = []
        async for event in agent.send_message(msg):
            results.append(event)

        if results:
            last = results[-1]
            if isinstance(last, A2AMessage):
                text_parts = [p.root.text for p in last.parts if p.root.kind == "text"]
                return " ".join(text_parts)

        return ""

    return StructuredTool.from_function(
        coroutine=_invoke_agent,
        name=agent.id,
        description=description,
        args_schema=args_schema,
    )
