import pytest

from sherma.entities.base import EntityBase
from sherma.exceptions import EntityNotFoundError, VersionNotFoundError
from sherma.registry.base import Registry, RegistryEntry


class DummyEntity(EntityBase):
    value: str = ""


class DummyRegistry(Registry[DummyEntity]):
    async def fetch(self, entry: RegistryEntry[DummyEntity]) -> DummyEntity:
        return DummyEntity(id=entry.id, version=entry.version, value="fetched")

    async def refresh(self, entry: RegistryEntry[DummyEntity]) -> None:
        instance = await self.fetch(entry)
        entry.instance = instance


@pytest.mark.asyncio
async def test_add_and_get_instance():
    reg = DummyRegistry()
    entity = DummyEntity(id="e1", version="1.0.0", value="hello")
    await reg.add(RegistryEntry(id="e1", version="1.0.0", instance=entity))
    result = await reg.get("e1", "==1.0.0")
    assert result.value == "hello"


@pytest.mark.asyncio
async def test_get_with_wildcard():
    reg = DummyRegistry()
    e1 = DummyEntity(id="e1", version="1.0.0", value="v1")
    e2 = DummyEntity(id="e1", version="2.0.0", value="v2")
    await reg.add(RegistryEntry(id="e1", version="1.0.0", instance=e1))
    await reg.add(RegistryEntry(id="e1", version="2.0.0", instance=e2))
    result = await reg.get("e1", "*")
    assert result.value == "v2"


@pytest.mark.asyncio
async def test_get_with_version_range():
    reg = DummyRegistry()
    e1 = DummyEntity(id="e1", version="1.0.0", value="v1.0")
    e2 = DummyEntity(id="e1", version="1.5.0", value="v1.5")
    e3 = DummyEntity(id="e1", version="2.0.0", value="v2.0")
    await reg.add(RegistryEntry(id="e1", version="1.0.0", instance=e1))
    await reg.add(RegistryEntry(id="e1", version="1.5.0", instance=e2))
    await reg.add(RegistryEntry(id="e1", version="2.0.0", instance=e3))
    result = await reg.get("e1", "==1.*")
    assert result.value == "v1.5"


@pytest.mark.asyncio
async def test_get_not_found():
    reg = DummyRegistry()
    with pytest.raises(EntityNotFoundError):
        await reg.get("missing")


@pytest.mark.asyncio
async def test_get_version_not_found():
    reg = DummyRegistry()
    e = DummyEntity(id="e1", version="1.0.0")
    await reg.add(RegistryEntry(id="e1", version="1.0.0", instance=e))
    with pytest.raises(VersionNotFoundError):
        await reg.get("e1", "==9.*")


@pytest.mark.asyncio
async def test_factory_resolution():
    reg = DummyRegistry()

    def make() -> DummyEntity:
        return DummyEntity(id="e1", version="1.0.0", value="from-factory")

    await reg.add(RegistryEntry(id="e1", version="1.0.0", factory=make))
    result = await reg.get("e1", "==1.0.0")
    assert result.value == "from-factory"


@pytest.mark.asyncio
async def test_async_factory_resolution():
    reg = DummyRegistry()

    async def make() -> DummyEntity:
        return DummyEntity(id="e1", version="1.0.0", value="async-factory")

    await reg.add(RegistryEntry(id="e1", version="1.0.0", factory=make))
    result = await reg.get("e1", "==1.0.0")
    assert result.value == "async-factory"


@pytest.mark.asyncio
async def test_remote_resolution():
    reg = DummyRegistry()
    await reg.add(RegistryEntry(id="e1", version="1.0.0", remote=True, url="http://x"))
    result = await reg.get("e1", "==1.0.0")
    assert result.value == "fetched"


@pytest.mark.asyncio
async def test_update_existing():
    reg = DummyRegistry()
    e1 = DummyEntity(id="e1", version="1.0.0", value="original")
    await reg.add(RegistryEntry(id="e1", version="1.0.0", instance=e1))

    e2 = DummyEntity(id="e1", version="1.0.0", value="updated")
    await reg.update(RegistryEntry(id="e1", version="1.0.0", instance=e2))
    result = await reg.get("e1", "==1.0.0")
    assert result.value == "updated"


@pytest.mark.asyncio
async def test_update_nonexistent_raises():
    reg = DummyRegistry()
    with pytest.raises(EntityNotFoundError):
        await reg.update(RegistryEntry(id="missing", version="1.0.0"))


@pytest.mark.asyncio
async def test_wildcard_version_entry():
    reg = DummyRegistry()
    e = DummyEntity(id="e1", version="*", value="wild")
    await reg.add(RegistryEntry(id="e1", version="*", instance=e))
    result = await reg.get("e1", "*")
    assert result.value == "wild"


@pytest.mark.asyncio
async def test_wildcard_entry_fallback():
    reg = DummyRegistry()
    e = DummyEntity(id="e1", version="*", value="fallback")
    await reg.add(RegistryEntry(id="e1", version="*", instance=e))
    result = await reg.get("e1", "==3.*")
    assert result.value == "fallback"
