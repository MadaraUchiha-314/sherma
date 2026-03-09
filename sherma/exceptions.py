class ShermaError(Exception):
    """Base exception for sherma framework."""


class EntityNotFoundError(ShermaError):
    """Raised when an entity is not found in the registry."""

    def __init__(self, entity_id: str, version: str = "*") -> None:
        self.entity_id = entity_id
        self.version = version
        super().__init__(f"Entity '{entity_id}' version '{version}' not found")


class VersionNotFoundError(ShermaError):
    """Raised when a specific version of an entity is not found."""

    def __init__(self, entity_id: str, version: str) -> None:
        self.entity_id = entity_id
        self.version = version
        super().__init__(f"Version '{version}' not found for entity '{entity_id}'")


class RegistryError(ShermaError):
    """Raised for registry operation failures."""


class SchemaValidationError(ShermaError):
    """Raised when data does not conform to a declared schema."""


class RemoteEntityError(ShermaError):
    """Raised when fetching a remote entity fails."""

    def __init__(self, entity_id: str, url: str, reason: str = "") -> None:
        self.entity_id = entity_id
        self.url = url
        msg = f"Failed to fetch remote entity '{entity_id}' from '{url}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class DeclarativeConfigError(ShermaError):
    """Raised when a declarative YAML config is invalid."""


class CelEvaluationError(ShermaError):
    """Raised when a CEL expression fails to evaluate."""


class GraphConstructionError(ShermaError):
    """Raised when building a LangGraph from declarative config fails."""
