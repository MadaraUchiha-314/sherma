"""Sidebar: LLM config, generated file viewer, download, and agent saving."""

from __future__ import annotations

import io
import os
import zipfile

import streamlit as st

from sherma.streamlit.state import default_model_for_provider


def render_sidebar() -> None:
    """Render the sidebar with LLM config, file viewer, and actions."""
    with st.sidebar:
        st.header("LLM Configuration")

        provider = st.selectbox(
            "Provider",
            options=["openai", "anthropic"],
            index=["openai", "anthropic"].index(st.session_state["llm_provider"]),
            key="provider_select",
        )
        if provider != st.session_state["llm_provider"]:
            st.session_state["llm_provider"] = provider
            st.session_state["model_name"] = ""

        default_model = default_model_for_provider(st.session_state["llm_provider"])
        model_name = st.text_input(
            "Model Name",
            value=st.session_state["model_name"] or default_model,
            key="model_input",
        )
        st.session_state["model_name"] = model_name

        # API key is never stored in session state to prevent it from
        # being persisted or sent to any Streamlit service.  The widget
        # value lives only in the browser and is read fresh each rerun
        # via get_effective_api_key().
        st.text_input(
            "API Key",
            value="",
            type="password",
            key="api_key_input",
            placeholder=(
                f"Or set {_env_var_for_provider(st.session_state['llm_provider'])}"
            ),
        )

        st.divider()

        # Generated files viewer
        st.header("Generated Files")
        files: dict[str, str] = st.session_state["generated_files"]
        if not files:
            st.info("No files generated yet. Chat with the designer!")
        else:
            for name, content in files.items():
                lang = _language_for_filename(name)
                with st.expander(name, expanded=False):
                    st.code(content, language=lang)

        if files:
            # Download all as ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, content in files.items():
                    zf.writestr(name, content)
            zip_buffer.seek(0)
            st.download_button(
                label="Download All Files",
                data=zip_buffer.getvalue(),
                file_name="agent_files.zip",
                mime="application/zip",
            )

        st.divider()

        # Save agent
        st.header("Save Agent")
        agent_name = st.text_input(
            "Agent Name", key="save_agent_name", placeholder="my-agent"
        )
        if st.button("Save Agent", disabled=not files or not agent_name):
            # Find the YAML file
            yaml_content = ""
            for name, content in files.items():
                if name.endswith(".yaml") or name.endswith(".yml"):
                    yaml_content = content
                    break
            if not yaml_content:
                st.error("No YAML config found in generated files.")
            else:
                agent_entry = {
                    "name": agent_name,
                    "yaml_content": yaml_content,
                    "files": dict(files),
                }
                st.session_state["created_agents"].append(agent_entry)
                st.session_state["agent_chat_messages"][agent_name] = []
                st.success(f"Agent '{agent_name}' saved!")


def _language_for_filename(filename: str) -> str:
    """Infer code language from filename extension."""
    if filename.endswith(".py"):
        return "python"
    if filename.endswith((".yaml", ".yml")):
        return "yaml"
    if filename.endswith(".json"):
        return "json"
    if filename.endswith(".md"):
        return "markdown"
    return "text"


def _env_var_for_provider(provider: str) -> str:
    """Return the environment variable name for the provider's API key."""
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    return "OPENAI_API_KEY"


def get_effective_api_key() -> str:
    """Return the API key from the widget or environment.

    The key is read directly from the Streamlit widget's internal state
    (keyed by ``"api_key_input"``).  It is **never** copied into
    ``st.session_state`` under a separate key, so Streamlit cannot
    persist or transmit it to any external service.
    """
    key: str = st.session_state.get("api_key_input", "")
    if key:
        return key
    provider = st.session_state.get("llm_provider", "openai")
    return os.environ.get(_env_var_for_provider(provider), "")
