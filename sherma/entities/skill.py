from typing import Any

from pydantic import BaseModel, Field

from sherma.entities.base import EntityBase
from sherma.entities.tool import Tool
from sherma.types import Markdown


class SkillFrontMatter(BaseModel):
    """YAML frontmatter for a skill definition."""

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, Any] | None = None
    allowed_tools: list[str] | None = None


class Skill(EntityBase):
    """A skill entity following the agentskills.io specification."""

    front_matter: SkillFrontMatter
    body: Markdown = ""
    scripts: list[Tool] = Field(default_factory=list)
    references: list[Markdown] = Field(default_factory=list)
    assets: list[Any] = Field(default_factory=list)
