"""Tests for tenant_id on EntityBase."""

from sherma.entities.base import DEFAULT_TENANT_ID, EntityBase


def test_entity_base_default_tenant() -> None:
    entity = EntityBase(id="x")
    assert entity.tenant_id == DEFAULT_TENANT_ID
    assert entity.tenant_id == "default"


def test_entity_base_custom_tenant() -> None:
    entity = EntityBase(id="x", tenant_id="acme")
    assert entity.tenant_id == "acme"
