"""Initialize and manage Streamlit session state."""

from __future__ import annotations

import uuid

import streamlit as st


def init_state() -> None:
    """Ensure all required session state keys exist."""
    defaults: dict[str, object] = {
        "llm_provider": "openai",
        "model_name": "",
        "designer_messages": [],
        "generated_files": {},
        "created_agents": [],
        "agent_chat_messages": {},
        "designer_thread_id": str(uuid.uuid4()),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def default_model_for_provider(provider: str) -> str:
    """Return a sensible default model name for a provider."""
    if provider == "anthropic":
        return "claude-sonnet-4-20250514"
    return "gpt-4o"
