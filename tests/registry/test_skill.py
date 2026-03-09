import pytest

from sherma.entities.skill import Skill, SkillFrontMatter
from sherma.registry.base import RegistryEntry
from sherma.registry.skill import SkillRegistry, _parse_skill_md


@pytest.mark.asyncio
async def test_skill_registry_local():
    reg = SkillRegistry()
    fm = SkillFrontMatter(name="greet", description="Greets people")
    s = Skill(id="greet", version="1.0.0", front_matter=fm)
    await reg.add(RegistryEntry(id="greet", version="1.0.0", instance=s))
    result = await reg.get("greet", "==1.0.0")
    assert result.front_matter.name == "greet"


def test_parse_skill_md():
    text = """---
name: test-skill
description: A test skill
license: MIT
---
# Body

Some content here.
"""
    skill = _parse_skill_md(text, "test", "1.0.0")
    assert skill.front_matter.name == "test-skill"
    assert skill.front_matter.description == "A test skill"
    assert skill.front_matter.license == "MIT"
    assert "Some content here." in skill.body


def test_parse_skill_md_no_frontmatter():
    text = "# Just a body"
    skill = _parse_skill_md(text, "test", "1.0.0")
    assert skill.front_matter.name == "test"
    assert skill.body == "# Just a body"
