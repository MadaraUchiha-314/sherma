from typing import Any

from sherma.entities.base import EntityBase


class Tool(EntityBase):
    """A tool entity wrapping a callable function."""

    function: Any
