from sherma.entities.agent.base import Agent
from sherma.entities.agent.remote import RemoteAgent
from sherma.exceptions import RemoteEntityError
from sherma.logging import get_logger
from sherma.registry.base import Registry, RegistryEntry

logger = get_logger(__name__)


class AgentRegistry(Registry[Agent]):
    """Registry for Agent entities.

    Remote agents use the A2A protocol.
    """

    async def fetch(self, entry: RegistryEntry[Agent]) -> Agent:
        url = self._require_url(entry)
        try:
            from a2a.client import A2AClient

            client = await A2AClient.get_client_from_agent_card_url(url)
            card = await client.get_card()
            return RemoteAgent(
                id=entry.id,
                version=entry.version,
                client=client,
                agent_card=card,
            )
        except ImportError as exc:
            raise RemoteEntityError(
                entry.id,
                url,
                "a2a-sdk is required for remote agents. "
                "Install with: pip install sherma[a2a]",
            ) from exc
