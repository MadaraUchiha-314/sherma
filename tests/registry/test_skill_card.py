"""Tests for SkillCardRegistry."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sherma.entities.skill_card import SkillCard
from sherma.registry.base import RegistryEntry
from sherma.registry.skill_card import SkillCardRegistry


@pytest.mark.asyncio
async def test_skill_card_registry_local():
    reg = SkillCardRegistry()
    card = SkillCard(
        id="my-skill",
        version="1.0.0",
        name="My Skill",
        description="A skill",
        base_uri="/path/to/skill",
        files=["SKILL.md"],
    )
    await reg.add(RegistryEntry(id="my-skill", version="1.0.0", instance=card))
    result = await reg.get("my-skill", "==1.0.0")
    assert result.name == "My Skill"
    assert result.files == ["SKILL.md"]


@pytest.mark.asyncio
async def test_skill_card_registry_fetch():
    reg = SkillCardRegistry()
    remote_data = {
        "name": "Remote Skill",
        "description": "Fetched remotely",
        "base_uri": "https://example.com/skill",
        "files": ["SKILL.md", "references/api.md"],
    }

    mock_response = MagicMock()
    mock_response.json.return_value = remote_data
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    async def fake_get_http_client(*a, **kw):
        return mock_client

    with patch(
        "sherma.registry.skill_card.get_http_client",
        side_effect=fake_get_http_client,
    ):
        entry = RegistryEntry(
            id="remote-skill",
            version="1.0.0",
            remote=True,
            url="https://example.com/skill/skill-card.json",
        )
        await reg.add(entry)
        result = await reg.get("remote-skill", "==1.0.0")

    assert result.name == "Remote Skill"
    assert result.base_uri == "https://example.com/skill"
    assert "references/api.md" in result.files


@pytest.mark.asyncio
async def test_skill_card_registry_version_matching():
    reg = SkillCardRegistry()
    card_v1 = SkillCard(
        id="s", version="1.0.0", name="V1", description="v1", base_uri="/v1"
    )
    card_v2 = SkillCard(
        id="s", version="2.0.0", name="V2", description="v2", base_uri="/v2"
    )
    await reg.add(RegistryEntry(id="s", version="1.0.0", instance=card_v1))
    await reg.add(RegistryEntry(id="s", version="2.0.0", instance=card_v2))

    result = await reg.get("s", ">=1.0.0")
    assert result.name == "V2"
