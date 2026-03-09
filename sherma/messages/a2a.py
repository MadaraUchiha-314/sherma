"""A2A message type helpers.

These helpers work with dict representations of A2A messages
to avoid hard dependency on a2a-sdk at runtime.
"""

from typing import Any


def make_text_part(text: str) -> dict[str, Any]:
    """Create a TextPart dict."""
    return {"kind": "text", "text": text}


def make_file_part(
    *, name: str, mime_type: str, data: str | None = None, uri: str | None = None
) -> dict[str, Any]:
    """Create a FilePart dict."""
    file_info: dict[str, Any] = {"name": name, "mimeType": mime_type}
    if data is not None:
        file_info["bytes"] = data
    if uri is not None:
        file_info["uri"] = uri
    return {"kind": "file", "file": file_info}


def make_data_part(data: dict[str, Any]) -> dict[str, Any]:
    """Create a DataPart dict."""
    return {"kind": "data", "data": data}


def make_message(
    *,
    role: str,
    parts: list[dict[str, Any]],
    message_id: str | None = None,
    task_id: str | None = None,
    context_id: str | None = None,
) -> dict[str, Any]:
    """Create an A2A message dict."""
    msg: dict[str, Any] = {"role": role, "parts": parts}
    if message_id is not None:
        msg["messageId"] = message_id
    if task_id is not None:
        msg["taskId"] = task_id
    if context_id is not None:
        msg["contextId"] = context_id
    return msg
