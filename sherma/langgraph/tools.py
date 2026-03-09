from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from sherma.entities.tool import Tool


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
