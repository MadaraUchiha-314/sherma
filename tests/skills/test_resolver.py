"""Tests for SkillResolver."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from sherma.entities.skill_card import SkillCard
from sherma.skills.resolver import SkillResolver


def _make_card(base_uri: str, files: list[str] | None = None) -> SkillCard:
    return SkillCard(
        id="test",
        version="1.0.0",
        name="Test",
        description="Test skill",
        base_uri=base_uri,
        files=files or [],
    )


def test_is_remote_http():
    resolver = SkillResolver(_make_card("http://example.com/skill"))
    assert resolver.is_remote() is True


def test_is_remote_https():
    resolver = SkillResolver(_make_card("https://example.com/skill"))
    assert resolver.is_remote() is True


def test_is_remote_local():
    resolver = SkillResolver(_make_card("/tmp/skill"))
    assert resolver.is_remote() is False


def test_resolve_path_local():
    resolver = SkillResolver(_make_card("/tmp/skill"))
    assert resolver.resolve_path("SKILL.md") == "/tmp/skill/SKILL.md"


def test_resolve_path_remote():
    resolver = SkillResolver(_make_card("https://example.com/skill"))
    assert resolver.resolve_path("SKILL.md") == "https://example.com/skill/SKILL.md"


def test_resolve_path_remote_trailing_slash():
    resolver = SkillResolver(_make_card("https://example.com/skill/"))
    assert resolver.resolve_path("SKILL.md") == "https://example.com/skill/SKILL.md"


@pytest.mark.asyncio
async def test_load_file_local():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Hello")

        resolver = SkillResolver(_make_card(str(skill_dir)))
        content = await resolver.load_file("SKILL.md")
        assert content == "# Hello"


@pytest.mark.asyncio
async def test_load_file_remote():
    mock_response = AsyncMock()
    mock_response.text = "# Remote Content"
    mock_response.raise_for_status = lambda: None

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    with patch("sherma.skills.resolver.get_http_client", return_value=mock_client):
        resolver = SkillResolver(_make_card("https://example.com/skill"))
        content = await resolver.load_file("SKILL.md")

    assert content == "# Remote Content"
    mock_client.get.assert_called_once_with("https://example.com/skill/SKILL.md")


def test_list_files_by_prefix():
    files = [
        "SKILL.md",
        "references/api.md",
        "references/guide.md",
        "assets/logo.png",
        "assets/diagram.svg",
    ]
    resolver = SkillResolver(_make_card("/tmp/skill", files))

    refs = resolver.list_files_by_prefix("references/")
    assert refs == ["references/api.md", "references/guide.md"]

    assets = resolver.list_files_by_prefix("assets/")
    assert assets == ["assets/logo.png", "assets/diagram.svg"]

    empty = resolver.list_files_by_prefix("scripts/")
    assert empty == []
