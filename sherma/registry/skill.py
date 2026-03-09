import yaml

from sherma.entities.skill import Skill, SkillFrontMatter
from sherma.http import get_http_client
from sherma.logging import get_logger
from sherma.registry.base import Registry, RegistryEntry

logger = get_logger(__name__)


def _parse_skill_md(text: str, entry_id: str, version: str) -> Skill:
    """Parse a SKILL.md file into a Skill entity.

    Expects YAML frontmatter between --- delimiters followed by markdown body.
    """
    parts = text.split("---", 2)
    if len(parts) < 3:
        return Skill(
            id=entry_id,
            version=version,
            front_matter=SkillFrontMatter(name=entry_id, description=""),
            body=text,
        )

    frontmatter_str = parts[1].strip()
    body = parts[2].strip()
    data = yaml.safe_load(frontmatter_str) or {}
    front_matter = SkillFrontMatter(**data)
    return Skill(
        id=entry_id,
        version=version,
        front_matter=front_matter,
        body=body,
    )


class SkillRegistry(Registry[Skill]):
    """Registry for Skill entities."""

    async def fetch(self, entry: RegistryEntry[Skill]) -> Skill:
        url = self._require_url(entry)
        client = await get_http_client()
        response = await client.get(url)
        response.raise_for_status()
        return _parse_skill_md(response.text, entry.id, entry.version)
