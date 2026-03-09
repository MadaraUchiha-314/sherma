import pytest

from sherma.entities.tool import Tool
from sherma.registry.base import RegistryEntry
from sherma.registry.tool import ToolRegistry


@pytest.mark.asyncio
async def test_tool_registry_local():
    reg = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    t = Tool(id="add", version="1.0.0", function=add)
    await reg.add(RegistryEntry(id="add", version="1.0.0", instance=t))
    result = await reg.get("add", "==1.0.0")
    assert result.function(1, 2) == 3
