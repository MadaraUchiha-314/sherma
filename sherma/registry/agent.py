from a2a.client import ClientConfig, ClientFactory

from sherma.entities.agent.base import Agent
from sherma.entities.agent.remote import RemoteAgent
from sherma.http import get_http_client
from sherma.logging import get_logger
from sherma.registry.base import Registry, RegistryEntry

logger = get_logger(__name__)


class AgentRegistry(Registry[Agent]):
    """Registry for Agent entities.

    Remote agents use the A2A protocol.
    """

    async def fetch(self, entry: RegistryEntry[Agent]) -> Agent:
        url = self._require_url(entry)
        httpx_client = await get_http_client()
        config = ClientConfig(httpx_client=httpx_client)
        client = await ClientFactory.connect(url, client_config=config)
        card = await client.get_card()
        return RemoteAgent(
            id=entry.id,
            version=entry.version,
            client=client,
            agent_card=card,
        )
