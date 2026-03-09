from pydantic import BaseModel, ConfigDict


class EntityBase(BaseModel):
    """Base class for all registry entities."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    version: str = "*"
