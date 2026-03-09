from sherma.entities.tool import Tool
from sherma.exceptions import RemoteEntityError
from sherma.logging import get_logger
from sherma.registry.base import Registry, RegistryEntry

logger = get_logger(__name__)


class ToolRegistry(Registry[Tool]):
    """Registry for Tool entities.

    Remote tools use MCP protocol.
    """

    async def fetch(self, entry: RegistryEntry[Tool]) -> Tool:
        url = self._require_url(entry)
        raise RemoteEntityError(
            entry.id,
            url,
            "MCP remote tool fetching requires mcp extra",
        )
