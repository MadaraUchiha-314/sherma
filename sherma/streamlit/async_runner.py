"""Bridge async DeclarativeAgent.send_message() to Streamlit's sync context."""

from __future__ import annotations

import asyncio
from typing import Any

import nest_asyncio
from a2a.types import Message

from sherma.langgraph.declarative.agent import DeclarativeAgent

nest_asyncio.apply()


def run_async(coro: Any) -> Any:
    """Run an async coroutine from synchronous Streamlit code."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def collect_events(agent: DeclarativeAgent, request: Message) -> list[Any]:
    """Collect all events from agent.send_message() into a list."""
    events: list[Any] = []
    async for event in agent.send_message(request):
        events.append(event)
    return events
