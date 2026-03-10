"""Tests for MCP tool loading from skill cards."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sherma.entities.skill_card import MCPServerDef, SkillCard
from sherma.skills.mcp import load_mcp_tools_from_skill


def _make_card_with_mcps() -> SkillCard:
    return SkillCard(
        id="test",
        version="1.0.0",
        name="Test",
        description="Test skill",
        base_uri="/tmp/skill",
        mcps={
            "server-1": MCPServerDef(
                id="server-1",
                url="http://localhost:8080",
                transport="streamable-http",
            )
        },
    )


@pytest.mark.asyncio
async def test_load_mcp_tools_empty():
    card = SkillCard(
        id="test",
        version="1.0.0",
        name="Test",
        description="Test",
        base_uri="/tmp",
    )
    tools = await load_mcp_tools_from_skill(card)
    assert tools == []


@pytest.mark.asyncio
async def test_load_mcp_tools_from_skill_card():
    mock_tool = MagicMock()
    mock_tool.name = "mcp-tool-1"

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[mock_tool])

    card = _make_card_with_mcps()

    with patch(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        return_value=mock_client,
    ):
        tools = await load_mcp_tools_from_skill(card)

    assert len(tools) == 1
