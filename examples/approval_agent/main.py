"""Approval agent — demonstrates message metadata routing via additional_kwargs.

A hook tags human responses with ``additional_kwargs["decision"]``, and CEL
edges inspect that metadata to approve or loop back for revision.

Usage:
    uv run python examples/approval_agent/main.py "Write a haiku about Python"

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
from langchain_core.messages import HumanMessage

from sherma.hooks.executor import BaseHookExecutor
from sherma.hooks.types import NodeExitContext
from sherma.http import get_http_client
from sherma.langgraph.declarative import DeclarativeAgent


class ApprovalTaggingHook(BaseHookExecutor):
    """Tags interrupt responses with ``additional_kwargs["decision"]``.

    After the interrupt node resumes, this hook inspects the user's reply.
    If the reply starts with "approve" (case-insensitive), the HumanMessage
    is tagged with ``additional_kwargs["decision"] = "approve"``.
    Otherwise it is tagged ``"revise"`` so the agent loops back for revision.

    This metadata is then available in CEL:
        state.messages[...]["additional_kwargs"]["decision"] == "approve"
    """

    async def node_exit(self, ctx: NodeExitContext) -> NodeExitContext | None:
        if ctx.node_type != "interrupt":
            return None

        messages = ctx.result.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, HumanMessage):
            return None

        content = str(last_msg.content).strip().lower()
        decision = "approve" if content.startswith("approve") else "revise"

        # Replace the message with one that carries the decision metadata
        tagged_msg = HumanMessage(
            content=last_msg.content,
            additional_kwargs={"decision": decision},
        )
        ctx.result["messages"] = [tagged_msg]
        return ctx


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python examples/approval_agent/main.py <query>")
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

    agent = DeclarativeAgent(
        id="approval-agent",
        version="1.0.0",
        yaml_path=Path(__file__).parent / "agent.yaml",
        http_async_client=http_client,
        hooks=[ApprovalTaggingHook()],
    )

    request = Message(
        message_id="user-1",
        parts=[Part(root=TextPart(text=sys.argv[1]))],
        role=Role.user,
    )

    print(f"Request: {sys.argv[1]}\n")

    msg_counter = 1
    while True:
        async for event in agent.send_message(request):
            if isinstance(event, TaskStatusUpdateEvent):
                if event.status.state == TaskState.input_required:
                    # Show the draft and ask for approval
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
            # Loop completed without break — no interrupt, done
            break


if __name__ == "__main__":
    asyncio.run(main())
