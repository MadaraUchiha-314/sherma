from sherma.entities.prompt import Prompt
from sherma.http import get_http_client
from sherma.logging import get_logger
from sherma.registry.base import Registry, RegistryEntry

logger = get_logger(__name__)


class PromptRegistry(Registry[Prompt]):
    """Registry for Prompt entities."""

    async def fetch(self, entry: RegistryEntry[Prompt]) -> Prompt:
        url = self._require_url(entry)
        client = await get_http_client()
        response = await client.get(url)
        response.raise_for_status()
        return Prompt(
            id=entry.id,
            version=entry.version,
            instructions=response.text,
        )
