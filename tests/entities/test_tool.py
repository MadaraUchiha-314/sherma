from sherma.entities.tool import Tool


def test_tool_creation():
    def add(a: int, b: int) -> int:
        return a + b

    t = Tool(id="add", version="1.0.0", function=add)
    assert t.function(1, 2) == 3
