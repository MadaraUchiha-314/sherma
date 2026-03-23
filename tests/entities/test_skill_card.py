"""Tests for SkillCard entity model."""

import pytest
from pydantic import ValidationError

from sherma.entities.skill_card import (
    LocalToolDef,
    MCPServerDef,
    SkillCard,
    SkillExtension,
)


def test_skill_card_minimal():
    card = SkillCard(
        id="my-skill",
        version="1.0.0",
        name="My Skill",
        description="Does something",
        base_uri="/path/to/skill",
    )
    assert card.id == "my-skill"
    assert card.name == "My Skill"
    assert card.files == []
    assert card.mcps == {}
    assert card.local_tools == {}
    assert card.extensions == []


def test_skill_card_full():
    card = SkillCard(
        id="my-skill",
        version="1.0.0",
        name="My Skill",
        description="Does something useful",
        base_uri="https://example.com/skills/my-skill",
        files=["SKILL.md", "assets/diagram.png", "references/api.md"],
        mcps={
            "my-mcp": MCPServerDef(
                id="my-mcp",
                version="1.0.0",
                url="http://localhost:8080",
                transport="streamable-http",
            )
        },
        local_tools={
            "my-tool": LocalToolDef(
                id="my-tool",
                version="1.0.0",
                import_path="my_module.tools.my_tool",
            )
        },
        extensions=[
            SkillExtension(
                uri="urn:skill:local_tools",
                description="Python tool references",
            ),
            SkillExtension(
                uri="urn:skill:mcps",
                description="MCP server definitions",
                required=True,
                params={"transport": "streamable-http"},
            ),
        ],
    )
    assert len(card.files) == 3
    assert "my-mcp" in card.mcps
    assert card.mcps["my-mcp"].transport == "streamable-http"
    assert "my-tool" in card.local_tools
    assert card.local_tools["my-tool"].import_path == "my_module.tools.my_tool"
    assert len(card.extensions) == 2
    assert card.extensions[0].uri == "urn:skill:local_tools"
    assert card.extensions[1].required is True
    assert card.extensions[1].params == {"transport": "streamable-http"}


def test_mcp_server_def():
    mcp = MCPServerDef(
        id="server-1",
        url="http://localhost:3000",
        transport="sse",
    )
    assert mcp.id == "server-1"
    assert mcp.version == "*"
    assert mcp.transport == "sse"


def test_mcp_server_def_missing_required():
    with pytest.raises(ValidationError):
        MCPServerDef(id="server-1", url="http://localhost:3000")  # type: ignore[call-arg]


def test_local_tool_def():
    tool = LocalToolDef(
        id="my-tool",
        import_path="examples.tools.get_weather",
    )
    assert tool.id == "my-tool"
    assert tool.version == "*"
    assert tool.import_path == "examples.tools.get_weather"


def test_local_tool_def_missing_required():
    with pytest.raises(ValidationError):
        LocalToolDef(id="my-tool")  # type: ignore[call-arg]


def test_skill_extension_minimal():
    ext = SkillExtension(uri="urn:skill:local_tools")
    assert ext.uri == "urn:skill:local_tools"
    assert ext.description is None
    assert ext.required is False
    assert ext.params is None


def test_skill_extension_full():
    ext = SkillExtension(
        uri="urn:skill:mcps",
        description="MCP server definitions",
        required=True,
        params={"transport": "streamable-http"},
    )
    assert ext.uri == "urn:skill:mcps"
    assert ext.description == "MCP server definitions"
    assert ext.required is True
    assert ext.params == {"transport": "streamable-http"}


def test_skill_extension_missing_uri():
    with pytest.raises(ValidationError):
        SkillExtension()  # type: ignore[call-arg]


def test_skill_card_missing_required():
    with pytest.raises(ValidationError):
        SkillCard(id="x", version="1.0.0", name="X")  # type: ignore[call-arg]


def test_skill_card_from_dict():
    data = {
        "id": "test",
        "version": "2.0.0",
        "name": "Test Skill",
        "description": "A test",
        "base_uri": "/tmp/test",
        "files": ["SKILL.md"],
        "mcps": {},
        "local_tools": {},
    }
    card = SkillCard.model_validate(data)
    assert card.id == "test"
    assert card.version == "2.0.0"
    assert card.files == ["SKILL.md"]
