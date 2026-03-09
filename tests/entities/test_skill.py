from sherma.entities.skill import Skill, SkillFrontMatter


def test_skill_creation():
    fm = SkillFrontMatter(name="greet", description="A greeting skill")
    s = Skill(id="greet", version="1.0.0", front_matter=fm, body="# Greet")
    assert s.front_matter.name == "greet"
    assert s.body == "# Greet"
    assert s.scripts == []
    assert s.references == []
    assert s.assets == []


def test_skill_front_matter_optional_fields():
    fm = SkillFrontMatter(
        name="test",
        description="desc",
        license="MIT",
        allowed_tools=["tool1"],
    )
    assert fm.license == "MIT"
    assert fm.allowed_tools == ["tool1"]
