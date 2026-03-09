"""Bidirectional lossless conversion between A2A and LangGraph messages.

Works with dict representations by default. When langchain-core is available,
produces actual LangChain message objects.
"""

from __future__ import annotations

from typing import Any

from sherma.logging import get_logger

logger = get_logger(__name__)

_METADATA_KEY = "a2a_metadata"


_langchain_messages: Any = None
_langchain_checked: bool = False


def _try_import_langchain() -> Any:
    """Try to import langchain_core.messages, return module or None (cached)."""
    global _langchain_messages, _langchain_checked
    if _langchain_checked:
        return _langchain_messages
    try:
        import langchain_core.messages as lc_messages

        _langchain_messages = lc_messages
    except ImportError:
        _langchain_messages = None
    _langchain_checked = True
    return _langchain_messages


def _a2a_part_to_content_block(part: dict[str, Any]) -> dict[str, Any] | str:
    """Convert a single A2A part to a LangGraph content block."""
    kind = part.get("kind")
    if kind == "text":
        return part.get("text", "")
    if kind == "file":
        return {"type": "file", "file": part.get("file", {})}
    if kind == "data":
        return {"type": "data", "data": part.get("data", {})}
    return {"type": "unknown", "raw": part}


def _content_block_to_a2a_part(block: dict[str, Any] | str) -> dict[str, Any]:
    """Convert a LangGraph content block back to an A2A part."""
    if isinstance(block, str):
        return {"kind": "text", "text": block}
    block_type = block.get("type")
    if block_type == "text":
        return {"kind": "text", "text": block.get("text", "")}
    if block_type == "file":
        return {"kind": "file", "file": block.get("file", {})}
    if block_type == "data":
        return {"kind": "data", "data": block.get("data", {})}
    return {"kind": "data", "data": block}


def a2a_to_langgraph(message: dict[str, Any]) -> list[Any]:
    """Convert an A2A message dict to a list of LangGraph messages.

    Preserves messageId, taskId, contextId in additional_kwargs.
    """
    lc = _try_import_langchain()

    role = message.get("role", "user")
    parts = message.get("parts", [])

    metadata: dict[str, Any] = {}
    for key in ("messageId", "taskId", "contextId"):
        if key in message:
            metadata[key] = message[key]

    # Build content
    if len(parts) == 1 and parts[0].get("kind") == "text":
        content: str | list[Any] = parts[0].get("text", "")
    else:
        content = [_a2a_part_to_content_block(p) for p in parts]

    additional_kwargs: dict[str, Any] = {}
    if metadata:
        additional_kwargs[_METADATA_KEY] = metadata

    if lc is not None:
        if role == "user":
            return [
                lc.HumanMessage(content=content, additional_kwargs=additional_kwargs)
            ]
        return [lc.AIMessage(content=content, additional_kwargs=additional_kwargs)]

    msg_type = "human" if role == "user" else "ai"
    return [
        {
            "type": msg_type,
            "content": content,
            "additional_kwargs": additional_kwargs,
        }
    ]


def langgraph_to_a2a(message: Any) -> dict[str, Any]:
    """Convert a LangGraph message (object or dict) to an A2A message dict.

    Restores messageId, taskId, contextId from additional_kwargs.
    """
    if isinstance(message, dict):
        msg_type = message.get("type", "ai")
        content = message.get("content", "")
        additional_kwargs = message.get("additional_kwargs", {})
    else:
        msg_type = message.type
        content = message.content
        additional_kwargs = getattr(message, "additional_kwargs", {})

    role = "user" if msg_type == "human" else "agent"

    # Build parts
    if isinstance(content, str):
        parts = [{"kind": "text", "text": content}]
    elif isinstance(content, list):
        parts = [_content_block_to_a2a_part(block) for block in content]
    else:
        parts = [{"kind": "text", "text": str(content)}]

    result: dict[str, Any] = {"role": role, "parts": parts}

    metadata = additional_kwargs.get(_METADATA_KEY, {})
    for key in ("messageId", "taskId", "contextId"):
        if key in metadata:
            result[key] = metadata[key]

    return result
