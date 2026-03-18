"""Shared chat display helpers for Streamlit."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_chat_history(messages: list[dict[str, Any]]) -> None:
    """Render a list of chat messages using st.chat_message.

    Each message dict has ``role`` (user/assistant) and ``content``.
    ``content`` can be a plain string or a list of content blocks where
    each block is either a string (text) or a dict with ``"type"``
    ``"image"`` and ``"data"`` (base64 bytes).
    """
    for msg in messages:
        with st.chat_message(msg["role"]):
            _render_content(msg["content"])


def _render_content(content: str | list[dict[str, Any] | str]) -> None:
    """Render a single message's content (text, images, or mixed)."""
    if isinstance(content, str):
        st.markdown(content)
        return
    for block in content:
        if isinstance(block, str):
            st.markdown(block)
        elif block.get("type") == "image":
            import base64

            st.image(base64.b64decode(block["data"]))
        else:
            st.markdown(str(block))
