from sherma.entities.base import EntityBase


class Prompt(EntityBase):
    """A prompt entity with instructions."""

    instructions: str
