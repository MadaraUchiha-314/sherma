"""Declarative CEL hooks agent — state updates after interrupt resume.

Same approval flow as examples/approval_agent, but the decision metadata is
turned into a ``response_type`` state field by an ``after_interrupt`` CEL hook
declared inline in ``agent.yaml`` — no Python hook class. CEL edges then route
on ``state.response_type``.

This mirrors the use case from issue #68 (declarative ``state_updates`` after
an interrupt resume).

Usage:
    uv run python examples/declarative_cel_hooks_agent/main.py "Write a haiku"

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
        print(
            "Usage: uv run python examples/declarative_cel_hooks_agent/main.py <query>"
        )
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

    # No hooks=[...] here — the CEL hook lives entirely in agent.yaml.
    agent = DeclarativeAgent(
        id="cel-hooks-approval-agent",
        version="1.0.0",
        yaml_path=Path(__file__).parent / "agent.yaml",
        http_async_client=http_client,
    )

    request = Message(
        message_id="user-1",
        parts=[Part(root=TextPart(text=sys.argv[1]))],
        role=Role.user,
    )

    print(f"Request: {sys.argv[1]}\n")
    print(
        "Tip: include the word 'approve' in your reply with "
        'additional_kwargs={"decision": "approve"} to accept.\n'
    )

    msg_counter = 1
    while True:
        async for event in agent.send_message(request):
            if isinstance(event, TaskStatusUpdateEvent):
                if event.status.state == TaskState.input_required:
                    if event.status.message:
                        for part in event.status.message.parts:
                            if part.root.kind == "text":
                                try:
                                    data = json.loads(part.root.text)
                                    draft = data.get("draft", part.root.text)
                                    hint = data.get("instructions", "")
                                    print(f"Draft: {draft}")
                                    if hint:
                                        print(f"\n{hint}")
                                except json.JSONDecodeError:
                                    print(part.root.text)
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
                        print(f"\nApproved: {part.root.text}")
        else:
            break


if __name__ == "__main__":
    asyncio.run(main())
