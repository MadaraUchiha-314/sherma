"""LangGraph/LangChain message type helpers.

These helpers work with dict representations to avoid
hard dependency on langchain-core at runtime.
"""

from typing import Any


def make_human_message(
    content: str | list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a HumanMessage-compatible dict."""
    return {"type": "human", "content": content, **kwargs}


def make_ai_message(
    content: str | list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Create an AIMessage-compatible dict."""
    return {"type": "ai", "content": content, **kwargs}
