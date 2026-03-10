from pydantic import BaseModel, ConfigDict

DEFAULT_TENANT_ID = "default"


class EntityBase(BaseModel):
    """Base class for all registry entities."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    version: str = "*"
    tenant_id: str = DEFAULT_TENANT_ID
