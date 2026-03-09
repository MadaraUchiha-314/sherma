from __future__ import annotations

from typing import TYPE_CHECKING

from sherma.entities.tool import Tool

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


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
    try:
        from langchain_core.tools import BaseTool as LCBaseTool
    except ImportError as e:
        msg = (
            "langchain-core is required for LangGraph tool conversion. "
            "Install with: pip install sherma[langgraph]"
        )
        raise ImportError(msg) from e

    if isinstance(tool.function, LCBaseTool):
        return tool.function

    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=tool.function,
        name=tool.id,
        description=f"Sherma tool: {tool.id}",
    )
