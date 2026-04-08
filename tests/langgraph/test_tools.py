from langchain_core.tools import BaseTool, tool

from sherma.entities.tool import Tool
from sherma.langgraph.tools import from_langgraph_tool, to_langgraph_tool


def test_from_langgraph_tool():
    @tool  # type: ignore[misc]
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    sherma_tool = from_langgraph_tool(add)
    assert sherma_tool.id == "add"
    assert sherma_tool.function is add


def test_to_langgraph_tool_passthrough():
    @tool  # type: ignore[misc]
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    sherma_tool = Tool(id="multiply", function=multiply)
    lc_tool = to_langgraph_tool(sherma_tool)
    assert lc_tool is multiply


def test_to_langgraph_tool_from_callable():
    def subtract(a: int, b: int) -> int:
        return a - b

    sherma_tool = Tool(id="subtract", function=subtract)
    lc_tool = to_langgraph_tool(sherma_tool)
    assert isinstance(lc_tool, BaseTool)
    assert lc_tool.name == "subtract"
