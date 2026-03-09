from sherma.entities.llm import LLM
from sherma.logging import get_logger
from sherma.registry.base import Registry, RegistryEntry

logger = get_logger(__name__)


class LLMRegistry(Registry[LLM]):
    """Registry for LLM entities."""

    async def fetch(self, entry: RegistryEntry[LLM]) -> LLM:
        url = self._require_url(entry)
        return LLM(
            id=entry.id,
            version=entry.version,
            model_name=url,
        )
