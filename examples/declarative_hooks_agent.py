"""Declarative weather agent with hooks — observability & guardrails.

Demonstrates two ways to register hook executors:
  1. Via the ``hooks`` constructor parameter (shown here)
  2. Via the YAML ``hooks:`` config section (also shown in the YAML)

Hook executors:
  - LoggingHook: prints node entry/exit, LLM calls, and tool calls
  - PromptGuardrailHook: appends a safety instruction to every LLM prompt

Usage:
    uv run python examples/declarative_hooks_agent.py "What is the weather in Paris?"

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

from examples.hooks import LoggingHook, PromptGuardrailHook
from sherma.http import get_http_client
from sherma.langgraph.declarative import DeclarativeAgent


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/declarative_hooks_agent.py <query>")
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

    # Hooks can be passed directly in the constructor.
    # They can also be declared in the YAML config via the hooks: section.
    # Constructor hooks run first, then YAML hooks, in registration order.
    agent = DeclarativeAgent(
        id="hooks-weather-agent",
        version="1.0.0",
        yaml_path=Path(__file__).parent / "declarative_hooks_agent.yaml",
        http_async_client=http_client,
        hooks=[LoggingHook(), PromptGuardrailHook()],
    )

    request = Message(
        message_id="user-1",
        parts=[Part(root=TextPart(text=sys.argv[1]))],
        role=Role.user,
    )

    print(f"Query: {sys.argv[1]}\n")
    async for event in agent.send_message(request):
        if isinstance(event, Message):
            for part in event.parts:
                if part.root.kind == "text":
                    print(f"\nAgent: {part.root.text}")


if __name__ == "__main__":
    asyncio.run(main())
