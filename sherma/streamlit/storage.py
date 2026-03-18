"""Module-level in-memory file store for Agent Designer tools.

The Streamlit app copies ``st.session_state["generated_files"]`` into the
module-level ``_store`` dict before each agent invocation, and copies it
back after.  The tools read/write ``_store`` directly.

A plain module-level dict is used instead of ``ContextVar`` because
LangGraph may execute synchronous tool functions in a thread-pool
worker, and ``ContextVar`` values are **not** inherited by threads.
"""

from __future__ import annotations

_store: dict[str, str] = {}


def get_store() -> dict[str, str]:
    """Return the current file store dict (filename -> content)."""
    return _store


def set_store(store: dict[str, str]) -> None:
    """Replace the current file store contents."""
    _store.clear()
    _store.update(store)
