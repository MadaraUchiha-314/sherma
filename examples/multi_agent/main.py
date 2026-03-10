"""Multi-agent example: Travel planner with weather sub-agent.

A supervisor agent delegates to a weather sub-agent (wrapped as a tool)
to provide travel recommendations with real-time weather data.

Usage:
    uv run python examples/multi_agent/main.py "Plan a trip to Paris"

Requires:
    - uv sync --extra examples
    - A secrets.json file at the project root with {"openai_api_key": "sk-..."}
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
from a2a.types import Message, Part, Role, TextPart

from sherma.http import get_http_client
from sherma.langgraph.declarative import DeclarativeAgent


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/multi_agent/main.py <query>")
        sys.exit(1)

    secrets_path = Path(__file__).resolve().parent.parent.parent / "secrets.json"
    if not secrets_path.exists():
        print(
            f"Error: {secrets_path} not found. "
            "Copy secrets.example.json and fill in your API key."
        )
        sys.exit(1)

    secrets = json.loads(secrets_path.read_text())
    api_key = secrets["openai_api_key"]

    http_client = await get_http_client(
        httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
        )
    )

    # The supervisor's YAML references the weather sub-agent via yaml_path.
    # The http_async_client is automatically forwarded to sub-agents.
    agent = DeclarativeAgent(
        id="travel-planner",
        version="1.0.0",
        yaml_path=Path(__file__).parent / "supervisor_agent.yaml",
        http_async_client=http_client,
    )

    request = Message(
        message_id="user-1",
        parts=[Part(root=TextPart(text=sys.argv[1]))],
        role=Role.user,
    )

    async for event in agent.send_message(request):
        if isinstance(event, Message):
            for part in event.parts:
                if part.root.kind == "text":
                    print(part.root.text)


if __name__ == "__main__":
    asyncio.run(main())
