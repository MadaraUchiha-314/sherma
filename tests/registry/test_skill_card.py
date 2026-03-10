"""Tests for SkillCard as an attribute on Skill."""

import pytest

from sherma.entities.skill import Skill, SkillFrontMatter
from sherma.entities.skill_card import SkillCard
from sherma.registry.base import RegistryEntry
from sherma.registry.skill import SkillRegistry


@pytest.mark.asyncio
async def test_skill_with_skill_card():
    reg = SkillRegistry()
    card = SkillCard(
        id="my-skill",
        version="1.0.0",
        name="My Skill",
        description="A skill",
        base_uri="/path/to/skill",
        files=["SKILL.md"],
    )
    skill = Skill(
        id="my-skill",
        version="1.0.0",
        front_matter=SkillFrontMatter(name="My Skill", description="A skill"),
        skill_card=card,
    )
    await reg.add(RegistryEntry(id="my-skill", version="1.0.0", instance=skill))
    result = await reg.get("my-skill", "==1.0.0")
    assert result.skill_card is not None
    assert result.skill_card.name == "My Skill"
    assert result.skill_card.files == ["SKILL.md"]


@pytest.mark.asyncio
async def test_skill_without_skill_card():
    reg = SkillRegistry()
    skill = Skill(
        id="simple-skill",
        version="1.0.0",
        front_matter=SkillFrontMatter(name="Simple", description="No card"),
    )
    await reg.add(RegistryEntry(id="simple-skill", version="1.0.0", instance=skill))
    result = await reg.get("simple-skill", "==1.0.0")
    assert result.skill_card is None
    assert result.front_matter.name == "Simple"


@pytest.mark.asyncio
async def test_skill_card_version_matching():
    reg = SkillRegistry()
    card_v1 = SkillCard(
        id="s", version="1.0.0", name="V1", description="v1", base_uri="/v1"
    )
    card_v2 = SkillCard(
        id="s", version="2.0.0", name="V2", description="v2", base_uri="/v2"
    )
    skill_v1 = Skill(
        id="s",
        version="1.0.0",
        front_matter=SkillFrontMatter(name="V1", description="v1"),
        skill_card=card_v1,
    )
    skill_v2 = Skill(
        id="s",
        version="2.0.0",
        front_matter=SkillFrontMatter(name="V2", description="v2"),
        skill_card=card_v2,
    )
    await reg.add(RegistryEntry(id="s", version="1.0.0", instance=skill_v1))
    await reg.add(RegistryEntry(id="s", version="2.0.0", instance=skill_v2))

    result = await reg.get("s", ">=1.0.0")
    assert result.skill_card is not None
    assert result.skill_card.name == "V2"
