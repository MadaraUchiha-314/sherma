"""Declarative weather agent — defined entirely via YAML.

Usage:
    uv run python examples/declarative_weather_agent.py "What is the weather?"

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
from a2a.types import Message, Part, Role, TaskState, TaskStatusUpdateEvent, TextPart

from sherma.http import get_http_client
from sherma.langgraph.declarative import DeclarativeAgent


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/declarative_weather_agent.py <query>")
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
        id="weather-agent",
        version="1.0.0",
        yaml_path=Path(__file__).parent / "declarative_weather_agent.yaml",
        http_async_client=http_client,
    )

    request = Message(
        message_id="user-1",
        parts=[Part(root=TextPart(text=sys.argv[1]))],
        role=Role.user,
    )

    msg_counter = 1
    while True:
        async for event in agent.send_message(request):
            if isinstance(event, TaskStatusUpdateEvent):
                if event.status.state == TaskState.input_required:
                    # Print the interrupt prompt
                    if event.status.message:
                        for part in event.status.message.parts:
                            if part.root.kind == "text":
                                print(part.root.text)
                    # Ask the user for input and send it back
                    user_input = input("> ")
                    msg_counter += 1
                    request = Message(
                        message_id=f"user-{msg_counter}",
                        parts=[Part(root=TextPart(text=user_input))],
                        role=Role.user,
                    )
                    break
            elif isinstance(event, Message):
                for part in event.parts:
                    if part.root.kind == "text":
                        print(part.root.text)
        else:
            # Loop completed without break — no interrupt, we're done
            break


if __name__ == "__main__":
    asyncio.run(main())
