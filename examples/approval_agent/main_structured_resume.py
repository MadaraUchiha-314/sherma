"""Approval agent — structured resume without the tagging hook.

This is a variant of ``main.py`` that demonstrates how passing a
``HumanMessage`` with ``additional_kwargs`` directly on interrupt resume
removes the need for the ``ApprovalTaggingHook``.

The same ``agent.yaml`` is reused — the CEL edge still routes on
``state.messages[...]["additional_kwargs"]["decision"]``. The difference
is that the client constructs that metadata up front and passes it
directly to ``Command(resume=...)``, and the interrupt node now preserves
the ``HumanMessage`` instead of stringifying it.

Usage:
    uv run python examples/approval_agent/main_structured_resume.py \
        "Write a haiku about Python"

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
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from sherma.http import get_http_client
from sherma.langgraph.declarative import DeclarativeAgent


async def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python examples/approval_agent/"
            "main_structured_resume.py <query>"
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
        httpx.AsyncClient(headers={"Authorization": f"Bearer {api_key}"})
    )

    # Note: no ApprovalTaggingHook — the client tags the resume message
    # itself and the interrupt node now preserves it as-is.
    agent = DeclarativeAgent(
        id="approval-agent",
        version="1.0.0",
        yaml_path=Path(__file__).parent / "agent.yaml",
        http_async_client=http_client,
    )

    graph = await agent.get_graph()
    config = {"configurable": {"thread_id": "structured-resume-demo"}}

    print(f"Request: {sys.argv[1]}\n")

    # Initial invocation — drafts and pauses at the interrupt node.
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=sys.argv[1])]},
        config=config,  # type: ignore[arg-type]
    )

    while result.get("__interrupt__"):
        # Display the draft from the interrupt payload.
        interrupt_value = result["__interrupt__"][0].value
        if isinstance(interrupt_value, dict):
            print(f"Draft: {interrupt_value.get('draft', interrupt_value)}")
            hint = interrupt_value.get("instructions", "")
            if hint:
                print(f"\n{hint}")
        else:
            print(interrupt_value)

        user_input = input("> ").strip()
        decision = "approve" if user_input.lower().startswith("approve") else "revise"

        # Pass a structured HumanMessage directly. With the fix from #63,
        # the interrupt node preserves this as-is — no after_interrupt hook
        # needed to attach `additional_kwargs["decision"]`.
        resume_msg = HumanMessage(
            content=user_input,
            additional_kwargs={"decision": decision},
        )
        result = await graph.ainvoke(
            Command(resume=[resume_msg]),
            config=config,  # type: ignore[arg-type]
        )

    final = result.get("messages", [])
    if final:
        print(f"\nApproved: {final[-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
