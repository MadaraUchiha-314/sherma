"""TenantRegistryManager: per-tenant singleton RegistryBundle instances."""

from __future__ import annotations

from sherma.entities.base import DEFAULT_TENANT_ID
from sherma.registry.bundle import RegistryBundle


class TenantRegistryManager:
    """Manages per-tenant singleton RegistryBundle instances."""

    def __init__(self) -> None:
        self._tenants: dict[str, RegistryBundle] = {}

    def get_bundle(self, tenant_id: str = DEFAULT_TENANT_ID) -> RegistryBundle:
        """Get or create the RegistryBundle for a tenant (singleton per tenant)."""
        if tenant_id not in self._tenants:
            self._tenants[tenant_id] = RegistryBundle(tenant_id=tenant_id)
        return self._tenants[tenant_id]

    def has_tenant(self, tenant_id: str) -> bool:
        """Check if a tenant exists."""
        return tenant_id in self._tenants

    def list_tenants(self) -> list[str]:
        """Return all tenant IDs."""
        return list(self._tenants.keys())

    def remove_tenant(self, tenant_id: str) -> None:
        """Remove a tenant and its registries."""
        self._tenants.pop(tenant_id, None)
