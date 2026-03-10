"""MCP tool loading from skill cards."""

from typing import Any

from langchain_core.tools import BaseTool

from sherma.entities.skill_card import SkillCard
from sherma.logging import get_logger

logger = get_logger(__name__)


async def load_mcp_tools_from_skill(skill_card: SkillCard) -> list[BaseTool]:
    """Load MCP tools defined in a skill card.

    Connects to each MCP server and returns LangChain BaseTool instances.
    """
    if not skill_card.mcps:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise ImportError(
            "langchain-mcp-adapters is required for MCP skill tools. "
            "Install with: uv add langchain-mcp-adapters"
        ) from exc

    connections: dict[str, Any] = {}
    for name, mcp_def in skill_card.mcps.items():
        connections[name] = {
            "url": mcp_def.url,
            "transport": mcp_def.transport,
        }

    client = MultiServerMCPClient(connections)
    return await client.get_tools()
