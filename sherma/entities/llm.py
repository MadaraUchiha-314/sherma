from sherma.entities.base import EntityBase


class LLM(EntityBase):
    """An LLM entity with a model name."""

    model_name: str
