"""Tests for TenantRegistryManager."""

import pytest

from sherma.entities.base import DEFAULT_TENANT_ID
from sherma.exceptions import EntityNotFoundError
from sherma.registry.base import RegistryEntry
from sherma.registry.tenant import TenantRegistryManager


def test_default_tenant_bundle() -> None:
    manager = TenantRegistryManager()
    bundle = manager.get_bundle()
    assert bundle.tenant_id == DEFAULT_TENANT_ID


def test_get_bundle_creates_singleton() -> None:
    manager = TenantRegistryManager()
    b1 = manager.get_bundle("t1")
    b2 = manager.get_bundle("t1")
    assert b1 is b2


def test_different_tenants_get_different_bundles() -> None:
    manager = TenantRegistryManager()
    b1 = manager.get_bundle("t1")
    b2 = manager.get_bundle("t2")
    assert b1 is not b2
    assert b1.tenant_id == "t1"
    assert b2.tenant_id == "t2"


@pytest.mark.anyio
async def test_tenant_isolation() -> None:
    from sherma.entities.base import EntityBase

    manager = TenantRegistryManager()
    b1 = manager.get_bundle("t1")
    b2 = manager.get_bundle("t2")

    await b1.llm_registry.add(
        RegistryEntry(
            id="my-llm",
            instance=EntityBase(id="my-llm"),
        )
    )

    result = await b1.llm_registry.get("my-llm")
    assert result.id == "my-llm"

    with pytest.raises(EntityNotFoundError):
        await b2.llm_registry.get("my-llm")


@pytest.mark.anyio
async def test_same_entity_id_different_tenants() -> None:
    from sherma.entities.prompt import Prompt

    manager = TenantRegistryManager()
    b1 = manager.get_bundle("t1")
    b2 = manager.get_bundle("t2")

    await b1.prompt_registry.add(
        RegistryEntry(
            id="greeting",
            instance=Prompt(id="greeting", instructions="Hello from t1"),
        )
    )
    await b2.prompt_registry.add(
        RegistryEntry(
            id="greeting",
            instance=Prompt(id="greeting", instructions="Hello from t2"),
        )
    )

    p1 = await b1.prompt_registry.get("greeting")
    p2 = await b2.prompt_registry.get("greeting")
    assert p1.instructions == "Hello from t1"
    assert p2.instructions == "Hello from t2"


def test_has_tenant() -> None:
    manager = TenantRegistryManager()
    assert not manager.has_tenant("t1")
    manager.get_bundle("t1")
    assert manager.has_tenant("t1")


def test_list_tenants() -> None:
    manager = TenantRegistryManager()
    assert manager.list_tenants() == []
    manager.get_bundle("t1")
    manager.get_bundle("t2")
    assert sorted(manager.list_tenants()) == ["t1", "t2"]


def test_remove_tenant() -> None:
    manager = TenantRegistryManager()
    manager.get_bundle("t1")
    assert manager.has_tenant("t1")
    manager.remove_tenant("t1")
    assert not manager.has_tenant("t1")


def test_remove_nonexistent_tenant() -> None:
    manager = TenantRegistryManager()
    manager.remove_tenant("nope")  # should not raise
