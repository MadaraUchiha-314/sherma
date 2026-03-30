"""Tests for LangGraph skill tools."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from sherma.entities.skill import Skill, SkillFrontMatter
from sherma.entities.skill_card import SkillCard
from sherma.langgraph.skill_tools import create_skill_tools
from sherma.registry.base import RegistryEntry
from sherma.registry.skill import SkillRegistry
from sherma.registry.tool import ToolRegistry


@pytest.fixture
def registries():
    return SkillRegistry(), ToolRegistry()


@pytest.fixture
def skill_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_path = Path(tmpdir) / "my-skill"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text(
            "---\nname: My Skill\ndescription: A test\n---\n# Instructions\nDo things."
        )
        (skill_path / "references").mkdir()
        (skill_path / "references" / "api.md").write_text("# API Reference")
        (skill_path / "assets").mkdir()
        (skill_path / "assets" / "logo.txt").write_text("LOGO")
        yield skill_path


@pytest_asyncio.fixture
async def setup_registries(registries, skill_dir):
    sk_reg, t_reg = registries
    card = SkillCard(
        id="my-skill",
        version="1.0.0",
        name="My Skill",
        description="A test skill",
        base_uri=str(skill_dir),
        files=["SKILL.md", "references/api.md", "assets/logo.txt"],
    )
    skill = Skill(
        id="my-skill",
        version="1.0.0",
        front_matter=SkillFrontMatter(name="My Skill", description="A test skill"),
        skill_card=card,
    )
    await sk_reg.add(RegistryEntry(id="my-skill", version="1.0.0", instance=skill))
    return sk_reg, t_reg


@pytest.mark.asyncio
async def test_list_skills(setup_registries):
    sk_reg, t_reg = setup_registries
    tools = create_skill_tools(sk_reg, t_reg)
    list_skills = tools[0]

    result = await list_skills.ainvoke({})
    assert len(result) == 1
    assert result[0]["id"] == "my-skill"
    assert result[0]["name"] == "My Skill"


@pytest.mark.asyncio
async def test_load_skill_md(setup_registries):
    sk_reg, t_reg = setup_registries
    tools = create_skill_tools(sk_reg, t_reg)
    load_skill_md = tools[1]

    with patch(
        "sherma.langgraph.skill_tools.load_mcp_tools_from_skill",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = await load_skill_md.ainvoke({"skill_id": "my-skill", "version": "*"})

    assert "Instructions" in result
    assert "Do things." in result

    # Skill should now be in skill registry with updated content
    skill = await sk_reg.get("my-skill", "==1.0.0")
    assert skill.front_matter.name == "My Skill"


@pytest.mark.asyncio
async def test_list_skill_resources(setup_registries):
    sk_reg, t_reg = setup_registries
    tools = create_skill_tools(sk_reg, t_reg)
    list_resources = tools[3]

    result = await list_resources.ainvoke({"skill_id": "my-skill", "version": "*"})
    assert result == ["references/api.md"]


@pytest.mark.asyncio
async def test_load_skill_resource(setup_registries):
    sk_reg, t_reg = setup_registries
    tools = create_skill_tools(sk_reg, t_reg)
    load_resource = tools[4]

    result = await load_resource.ainvoke(
        {"skill_id": "my-skill", "resource_path": "references/api.md", "version": "*"}
    )
    assert "API Reference" in result


@pytest.mark.asyncio
async def test_list_skill_assets(setup_registries):
    sk_reg, t_reg = setup_registries
    tools = create_skill_tools(sk_reg, t_reg)
    list_assets = tools[5]

    result = await list_assets.ainvoke({"skill_id": "my-skill", "version": "*"})
    assert result == ["assets/logo.txt"]


@pytest.mark.asyncio
async def test_load_skill_asset(setup_registries):
    sk_reg, t_reg = setup_registries
    tools = create_skill_tools(sk_reg, t_reg)
    load_asset = tools[6]

    result = await load_asset.ainvoke(
        {"skill_id": "my-skill", "asset_path": "assets/logo.txt", "version": "*"}
    )
    assert result == "LOGO"


@pytest.mark.asyncio
async def test_unload_skill(setup_registries):
    sk_reg, t_reg = setup_registries
    tools = create_skill_tools(sk_reg, t_reg)
    load_skill_md = tools[1]
    unload_skill = tools[2]

    # First load the skill
    with patch(
        "sherma.langgraph.skill_tools.load_mcp_tools_from_skill",
        new_callable=AsyncMock,
        return_value=[],
    ):
        await load_skill_md.ainvoke({"skill_id": "my-skill", "version": "*"})

    # Now unload it
    result = await unload_skill.ainvoke({"skill_id": "my-skill", "version": "*"})
    assert "unloaded" in result.lower()


@pytest.mark.asyncio
async def test_unload_skill_no_skill_card(setup_registries):
    """Unloading a skill with no skill card should succeed gracefully."""
    sk_reg, t_reg = setup_registries

    # Add a skill without a skill card
    no_card_skill = Skill(
        id="no-card",
        version="1.0.0",
        front_matter=SkillFrontMatter(name="No Card", description="No card"),
    )
    await sk_reg.add(
        RegistryEntry(id="no-card", version="1.0.0", instance=no_card_skill)
    )

    tools = create_skill_tools(sk_reg, t_reg)
    unload_skill = tools[2]

    result = await unload_skill.ainvoke({"skill_id": "no-card", "version": "*"})
    assert "unloaded" in result.lower()
    assert "no tools" in result.lower()


@pytest.mark.asyncio
async def test_create_skill_tools_returns_seven(registries):
    sk_reg, t_reg = registries
    tools = create_skill_tools(sk_reg, t_reg)
    assert len(tools) == 7
    names = [t.name for t in tools]
    assert "list_skills" in names
    assert "load_skill_md" in names
    assert "unload_skill" in names
    assert "list_skill_resources" in names
    assert "load_skill_resource" in names
    assert "list_skill_assets" in names
    assert "load_skill_asset" in names
