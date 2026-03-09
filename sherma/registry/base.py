from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from sherma.entities.base import EntityBase
from sherma.exceptions import (
    EntityNotFoundError,
    RemoteEntityError,
    VersionNotFoundError,
)
from sherma.logging import get_logger
from sherma.types import Protocol
from sherma.version import WILDCARD, find_best_match

logger = get_logger(__name__)

T = TypeVar("T", bound=EntityBase)


class RegistryEntry(BaseModel, Generic[T]):
    """An entry in the registry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    version: str = WILDCARD
    remote: bool = False
    instance: T | None = None
    factory: Callable[[], T | Awaitable[T]] | None = None
    url: str | None = None
    protocol: Protocol | None = None


class Registry(ABC, Generic[T]):
    """Generic registry for entities."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, RegistryEntry[T]]] = {}

    async def add(self, entry: RegistryEntry[T]) -> None:
        """Add an entry to the registry."""
        if entry.id not in self._entries:
            self._entries[entry.id] = {}
        self._entries[entry.id][entry.version] = entry
        logger.debug("Added %s/%s to registry", entry.id, entry.version)

    async def update(self, entry: RegistryEntry[T]) -> None:
        """Update an existing entry."""
        if entry.id not in self._entries:
            raise EntityNotFoundError(entry.id, entry.version)
        if entry.version not in self._entries[entry.id]:
            raise VersionNotFoundError(entry.id, entry.version)
        self._entries[entry.id][entry.version] = entry

    async def get(self, entity_id: str, version: str = WILDCARD) -> T:
        """Resolve and return an entity by id and version specifier."""
        if entity_id not in self._entries:
            raise EntityNotFoundError(entity_id, version)

        versions = list(self._entries[entity_id].keys())
        matched_version = find_best_match(versions, version)
        if matched_version is None:
            raise VersionNotFoundError(entity_id, version)

        entry = self._entries[entity_id][matched_version]
        return await self._resolve(entry)

    async def _resolve(self, entry: RegistryEntry[T]) -> T:
        """Resolve an entry to an entity instance."""
        if entry.instance is not None:
            return entry.instance

        if entry.factory is not None:
            result = entry.factory()
            if isinstance(result, Awaitable):
                instance = await result
            else:
                instance = result  # type: ignore[assignment]
            entry.instance = instance
            return instance

        if entry.remote and entry.url is not None:
            instance = await self.fetch(entry)
            entry.instance = instance
            return instance

        raise EntityNotFoundError(entry.id, entry.version)

    @staticmethod
    def _require_url(entry: RegistryEntry[T]) -> str:
        """Validate that entry has a URL, raising if not."""
        if entry.url is None:
            raise RemoteEntityError(entry.id, "", "No URL provided")
        return entry.url

    @abstractmethod
    async def fetch(self, entry: RegistryEntry[T]) -> T:
        """Fetch a remote entity. Implemented by subclasses."""
        ...

    async def refresh(self, entry: RegistryEntry[T]) -> None:
        """Refresh a remote entity by re-fetching it."""
        instance = await self.fetch(entry)
        entry.instance = instance
