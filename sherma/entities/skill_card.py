"""SkillCard entity for agentskills.io skill discovery."""

from typing import Any

from pydantic import BaseModel, Field

from sherma.entities.base import EntityBase


class MCPServerDef(BaseModel):
    """MCP server definition within a skill card."""

    id: str
    version: str = "*"
    url: str
    transport: str  # "stdio" | "sse" | "streamable-http"


class LocalToolDef(BaseModel):
    """Local tool definition within a skill card."""

    id: str
    version: str = "*"
    import_path: str  # e.g. "examples.tools.get_weather"


class SkillCard(EntityBase):
    """A skill card for progressive skill discovery.

    Contains metadata for listing, file references for loading,
    and MCP/local tool definitions for tool attachment.
    """

    name: str
    description: str
    base_uri: str
    files: list[str] = Field(default_factory=list)
    mcps: dict[str, MCPServerDef] = Field(default_factory=dict)
    local_tools: dict[str, LocalToolDef] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)
