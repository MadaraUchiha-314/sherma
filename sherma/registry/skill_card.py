"""Registry for SkillCard entities."""

from sherma.entities.skill_card import SkillCard
from sherma.http import get_http_client
from sherma.logging import get_logger
from sherma.registry.base import Registry, RegistryEntry

logger = get_logger(__name__)


class SkillCardRegistry(Registry[SkillCard]):
    """Registry for SkillCard entities."""

    async def fetch(self, entry: RegistryEntry[SkillCard]) -> SkillCard:
        url = self._require_url(entry)
        client = await get_http_client()
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        return SkillCard(
            id=entry.id,
            version=entry.version,
            **{k: v for k, v in data.items() if k not in ("id", "version")},
        )
