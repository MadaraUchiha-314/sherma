import pytest

from sherma.entities.tool import Tool
from sherma.exceptions import EntityNotFoundError
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


@pytest.mark.asyncio
async def test_tool_registry_remove():
    reg = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    t = Tool(id="add", version="1.0.0", function=add)
    await reg.add(RegistryEntry(id="add", version="1.0.0", instance=t))
    await reg.remove("add", "==1.0.0")

    with pytest.raises(EntityNotFoundError):
        await reg.get("add", "==1.0.0")


@pytest.mark.asyncio
async def test_tool_registry_remove_wildcard():
    reg = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    t1 = Tool(id="add", version="1.0.0", function=add)
    t2 = Tool(id="add", version="2.0.0", function=add)
    await reg.add(RegistryEntry(id="add", version="1.0.0", instance=t1))
    await reg.add(RegistryEntry(id="add", version="2.0.0", instance=t2))

    await reg.remove("add")  # wildcard removes all versions

    with pytest.raises(EntityNotFoundError):
        await reg.get("add")


@pytest.mark.asyncio
async def test_tool_registry_remove_nonexistent():
    reg = ToolRegistry()

    with pytest.raises(EntityNotFoundError):
        await reg.remove("nonexistent")
