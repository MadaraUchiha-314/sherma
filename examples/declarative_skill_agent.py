"""Declarative skill agent — discovers and uses skills via YAML config.

Usage:
    uv run python examples/declarative_skill_agent.py "What's the weather in Tokyo?"

Requires:
    - uv sync --extra examples
    - A secrets.json file at the project root with {"openai_api_key": "sk-..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx
from a2a.types import Message, Part, Role, TextPart

from sherma.http import get_http_client
from sherma.langgraph.declarative import DeclarativeAgent


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Silence noisy loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("celpy").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    if len(sys.argv) < 2:
        print("Usage: uv run python examples/declarative_skill_agent.py <query>")
        sys.exit(1)

    secrets_path = Path(__file__).resolve().parent.parent / "secrets.json"
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

    agent = DeclarativeAgent(
        id="skill-agent",
        version="1.0.0",
        yaml_path=Path(__file__).parent / "declarative_skill_agent.yaml",
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
