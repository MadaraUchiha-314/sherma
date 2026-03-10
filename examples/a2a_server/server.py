"""A2A JSON-RPC server exposing a declarative skill agent.

Starts an A2A-compliant JSON-RPC server on port 3000 that wraps
the declarative skill agent (weather example). Clients can interact
with the agent using standard A2A protocol messages over HTTP.

Usage:
    uv run python examples/a2a_server/server.py

Requires:
    - uv sync --extra examples
    - A secrets.json file at the project root with
      {"openai_api_key": "sk-..."}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import uvicorn
from a2a.server.apps.jsonrpc import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from sherma.a2a import ShermaAgentExecutor
from sherma.http import get_http_client
from sherma.langgraph.declarative import DeclarativeAgent

PORT = 3000
HOST = "0.0.0.0"


def _load_api_key() -> str:
    secrets_path = Path(__file__).resolve().parent.parent.parent / "secrets.json"
    if not secrets_path.exists():
        print(
            f"Error: {secrets_path} not found. "
            "Copy secrets.example.json and fill in your API key."
        )
        sys.exit(1)
    return json.loads(secrets_path.read_text())["openai_api_key"]


def _build_agent_card() -> AgentCard:
    return AgentCard(
        name="Weather Skill Agent",
        description=(
            "A general-purpose assistant that discovers and uses "
            "skills to answer weather queries."
        ),
        url=f"http://{HOST}:{PORT}/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="weather-lookup",
                name="Weather Lookup",
                description="Look up current weather for any city.",
                tags=["weather", "forecast"],
                examples=[
                    "What's the weather in Tokyo?",
                    "Is it raining in London?",
                ],
            ),
        ],
    )


async def _create_app() -> A2AStarletteApplication:
    api_key = _load_api_key()

    http_client = await get_http_client(
        httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
        )
    )

    agent = DeclarativeAgent(
        id="skill-agent",
        version="1.0.0",
        yaml_path=(
            Path(__file__).parent.parent / "declarative_skill_agent" / "agent.yaml"
        ),
        http_async_client=http_client,
    )

    executor = ShermaAgentExecutor(agent=agent)
    agent_card = _build_agent_card()

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )


async def main() -> None:
    a2a_app = await _create_app()
    starlette_app = a2a_app.build()

    config = uvicorn.Config(
        starlette_app,
        host=HOST,
        port=PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
