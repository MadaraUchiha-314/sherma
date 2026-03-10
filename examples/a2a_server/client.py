"""A2A client that sends a message to the weather agent server.

Connects to the A2A server started by server.py and sends a weather
query. Prints the agent's response.

Usage:
    # First start the server in one terminal:
    #   uv run python examples/a2a_server/server.py
    #
    # Then run this client in another terminal:
    uv run python examples/a2a_server/client.py "What is the weather in Tokyo?"
"""

from __future__ import annotations

import asyncio
import sys

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

SERVER_URL = "http://localhost:3000"

# Agent may take a while (multiple LLM calls), so use a generous timeout
REQUEST_TIMEOUT = 120.0


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/a2a_server/client.py <query>")
        sys.exit(1)

    query = sys.argv[1]

    # Resolve the agent card and display agent info
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=SERVER_URL)
        agent_card = await resolver.get_agent_card()

    print(f"Connected to: {agent_card.name}")
    print(f"Description:  {agent_card.description}")
    print(f"Skills:       {[s.name for s in agent_card.skills]}")
    print()

    # Connect with a long-lived httpx client for streaming
    httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT))
    config = ClientConfig(httpx_client=httpx_client)
    client = await ClientFactory.connect(SERVER_URL, client_config=config)

    # Send the user's query
    request = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=query))],
        message_id="client-1",
    )

    print(f"User: {query}")
    print()

    async for event in client.send_message(request):
        if isinstance(event, Message):
            for part in event.parts:
                if part.root.kind == "text":
                    print(f"Agent: {part.root.text}")
        else:
            # (Task, UpdateEvent) tuple
            _task, update = event
            if update is not None:
                status = getattr(update, "status", None)
                if status and status.message:
                    for part in status.message.parts:
                        if part.root.kind == "text":
                            print(f"Agent: {part.root.text}")


if __name__ == "__main__":
    asyncio.run(main())
