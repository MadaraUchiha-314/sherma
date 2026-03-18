"""Main Streamlit entry point for the Sherma Agent Designer."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from a2a.types import (
    FilePart,
    FileWithBytes,
    Message,
    Part,
    Role,
    TaskStatusUpdateEvent,
    TextPart,
)

from sherma.langgraph.declarative.agent import DeclarativeAgent
from sherma.streamlit.async_runner import collect_events, run_async
from sherma.streamlit.components.chat_ui import render_chat_history
from sherma.streamlit.components.sidebar import (
    get_effective_api_key,
    render_sidebar,
)
from sherma.streamlit.state import init_state
from sherma.streamlit.storage import get_store, set_store

_AGENT_YAML_PATH = Path(__file__).parent / "agent_designer" / "agent.yaml"

_IMAGE_TYPES = ["png", "jpg", "jpeg", "gif", "webp"]


def _build_designer_yaml(provider: str, model_name: str) -> str:
    """Load the bundled agent.yaml and override LLM provider/model."""
    raw = _AGENT_YAML_PATH.read_text()
    data = yaml.safe_load(raw)
    if data.get("llms"):
        data["llms"][0]["provider"] = provider
        data["llms"][0]["model_name"] = model_name
    return yaml.dump(data, default_flow_style=False)


def _extract_text_from_message(msg: Message) -> str:
    """Pull text from an A2A Message's parts."""
    texts = []
    for part in msg.parts:
        if part.root.kind == "text":
            texts.append(part.root.text)
    return "\n".join(texts)


def _extract_response_text(events: list[object]) -> str:
    """Extract text content from agent response events.

    Handles both completed responses (``Message``) and interrupts
    (``TaskStatusUpdateEvent`` with ``input_required``).
    """
    for event in events:
        if isinstance(event, TaskStatusUpdateEvent):
            status_msg = event.status.message
            if status_msg is not None:
                text = _extract_text_from_message(status_msg)
                if text:
                    return text
        if isinstance(event, Message):
            text = _extract_text_from_message(event)
            if text:
                return text
    return "*(No response)*"


def _set_api_key_env(provider: str, api_key: str) -> None:
    """Set the API key as an environment variable for the LLM provider.

    The LangChain chat model constructors (ChatOpenAI, ChatAnthropic)
    read from these env vars natively.  This keeps the key local to the
    process and avoids placing it on any network-bound HTTP client.
    """
    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    os.environ[env_var] = api_key


def _uploaded_files_to_parts(
    uploaded_files: list[Any],
) -> tuple[list[Part], list[dict[str, Any]]]:
    """Convert Streamlit UploadedFile objects to A2A Parts and history blocks.

    Returns (a2a_parts, history_blocks) where history_blocks store
    base64-encoded image data for rendering in the chat UI.
    """
    a2a_parts: list[Part] = []
    history_blocks: list[dict[str, Any]] = []
    for uf in uploaded_files:
        raw_bytes = uf.read()
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        mime = uf.type or "image/png"
        a2a_parts.append(
            Part(
                root=FilePart(
                    file=FileWithBytes(
                        bytes=b64,
                        mime_type=mime,
                        name=uf.name,
                    ),
                )
            )
        )
        history_blocks.append({"type": "image", "data": b64})
    return a2a_parts, history_blocks


def _build_multimodal_content(
    text: str,
    image_blocks: list[dict[str, Any]],
) -> str | list[dict[str, Any] | str]:
    """Build a content value for the chat history.

    Returns a plain string when there are no images, or a list of mixed
    text/image blocks for multimodal messages.
    """
    if not image_blocks:
        return text
    content: list[dict[str, Any] | str] = []
    if text:
        content.append(text)
    content.extend(image_blocks)
    return content


def _run_designer(user_input: str, image_parts: list[Part] | None = None) -> str:
    """Send a message to the Agent Designer and return the response."""
    api_key = get_effective_api_key()
    if not api_key:
        return "Please provide an API key in the sidebar."

    provider = st.session_state["llm_provider"]
    model_name = st.session_state["model_name"]

    _set_api_key_env(provider, api_key)

    modified_yaml = _build_designer_yaml(provider, model_name)

    agent = DeclarativeAgent(
        id="agent-designer",
        version="1.0.0",
        yaml_content=modified_yaml,
    )

    set_store(dict(st.session_state["generated_files"]))

    messages = st.session_state["designer_messages"]

    # Build A2A parts: text + optional images
    parts: list[Part] = []
    if user_input:
        parts.append(Part(root=TextPart(text=user_input)))
    if image_parts:
        parts.extend(image_parts)
    if not parts:
        parts.append(Part(root=TextPart(text="")))

    request = Message(
        message_id=f"user-{len(messages)}",
        parts=parts,
        role=Role.user,
        context_id=st.session_state["designer_thread_id"],
    )

    events = run_async(collect_events(agent, request))
    response_text = _extract_response_text(events)

    # Sync store back — tools mutated the module-level dict directly
    st.session_state["generated_files"] = dict(get_store())

    return response_text


def _run_created_agent(
    agent_entry: dict[str, object],
    user_input: str,
    image_parts: list[Part] | None = None,
) -> str:
    """Send a message to a user-created agent and return the response."""
    api_key = get_effective_api_key()
    if not api_key:
        return "Please provide an API key in the sidebar."

    provider = st.session_state["llm_provider"]
    model_name = st.session_state["model_name"]
    yaml_content = str(agent_entry["yaml_content"])

    _set_api_key_env(provider, api_key)

    try:
        data = yaml.safe_load(yaml_content)
        if data.get("llms"):
            data["llms"][0]["provider"] = provider
            data["llms"][0]["model_name"] = model_name
        yaml_content = yaml.dump(data, default_flow_style=False)
    except yaml.YAMLError:
        pass

    try:
        data = yaml.safe_load(yaml_content)
        agents = data.get("agents", {})
        agent_id = next(iter(agents)) if agents else "agent"
    except Exception:
        agent_id = "agent"

    agent_name = str(agent_entry["name"])

    agent = DeclarativeAgent(
        id=agent_id,
        version="1.0.0",
        yaml_content=yaml_content,
    )

    thread_id = f"chat-{agent_name}"

    chat_msgs = st.session_state["agent_chat_messages"]
    msg_count = len(chat_msgs.get(agent_name, []))

    parts: list[Part] = []
    if user_input:
        parts.append(Part(root=TextPart(text=user_input)))
    if image_parts:
        parts.extend(image_parts)
    if not parts:
        parts.append(Part(root=TextPart(text="")))

    request = Message(
        message_id=f"user-{msg_count}",
        parts=parts,
        role=Role.user,
        context_id=thread_id,
    )

    try:
        events = run_async(collect_events(agent, request))
        return _extract_response_text(events)
    except Exception as exc:
        return f"Error: {exc}"


def _render_input_area(
    *,
    chat_key: str,
    upload_key: str,
) -> tuple[str, list[Any], bool]:
    """Render a chat input + image uploader.

    Returns ``(text, files, submitted)`` where *submitted* is True only
    when the user presses Enter in the chat input.  Images are staged
    in the uploader until the user sends a message — uploading alone
    does not trigger a send (which would cause an infinite rerun loop
    because ``st.file_uploader`` retains its value across reruns).
    """
    uploaded = st.file_uploader(
        "Attach images",
        type=_IMAGE_TYPES,
        accept_multiple_files=True,
        key=upload_key,
        label_visibility="collapsed",
    )
    text = st.chat_input("Type a message...", key=chat_key)
    # Only treat as submitted when the user presses Enter.
    submitted = text is not None and text != ""
    return text or "", uploaded or [], submitted


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        layout="wide",
        page_title="Sherma Agent Designer",
        page_icon="🏗️",
    )

    init_state()
    render_sidebar()

    tab_designer, tab_chat = st.tabs(["Agent Designer", "Chat With Agent"])

    # --- Tab 1: Agent Designer ---
    with tab_designer:
        st.title("Agent Designer")
        st.caption(
            "Describe the agent you want to create. "
            "The designer will generate YAML configs, "
            "tools, and prompts."
        )

        render_chat_history(st.session_state["designer_messages"])

        user_input, uploaded_files, submitted = _render_input_area(
            chat_key="designer_input",
            upload_key="designer_upload",
        )

        if submitted:
            image_parts, image_blocks = _uploaded_files_to_parts(uploaded_files)
            content = _build_multimodal_content(user_input, image_blocks)

            with st.chat_message("user"):
                from sherma.streamlit.components.chat_ui import (
                    _render_content,
                )

                _render_content(content)

            st.session_state["designer_messages"].append(
                {"role": "user", "content": content}
            )

            with st.chat_message("assistant"):
                with st.spinner("Designing..."):
                    response = _run_designer(user_input, image_parts or None)
                st.markdown(response)

            st.session_state["designer_messages"].append(
                {"role": "assistant", "content": response}
            )

            st.rerun()

    # --- Tab 2: Chat With Agent ---
    with tab_chat:
        st.title("Chat With Agent")

        created_agents: list[dict[str, object]] = st.session_state["created_agents"]

        if not created_agents:
            st.info(
                "No agents created yet. "
                "Use the Agent Designer tab to create one, "
                "then save it using the sidebar."
            )
        else:
            agent_names = [str(a["name"]) for a in created_agents]
            selected_name = st.selectbox(
                "Select Agent",
                options=agent_names,
                key="agent_select",
            )

            if selected_name:
                selected_agent = next(
                    a for a in created_agents if a["name"] == selected_name
                )

                try:
                    data = yaml.safe_load(str(selected_agent["yaml_content"]))
                    has_custom_tools = bool(data.get("tools"))
                except Exception:
                    has_custom_tools = False

                if has_custom_tools:
                    st.warning(
                        "This agent has custom tools with "
                        "import_path references. "
                        "Custom tools require the Python "
                        "modules to be available locally. "
                        "Download the files and run locally "
                        "for full functionality."
                    )

                chat_key = selected_name
                if chat_key not in st.session_state["agent_chat_messages"]:
                    st.session_state["agent_chat_messages"][chat_key] = []

                messages = st.session_state["agent_chat_messages"][chat_key]
                render_chat_history(messages)

                user_input, uploaded_files, submitted = _render_input_area(
                    chat_key="agent_chat_input",
                    upload_key="agent_chat_upload",
                )

                if submitted:
                    image_parts, image_blocks = _uploaded_files_to_parts(uploaded_files)
                    content = _build_multimodal_content(user_input, image_blocks)

                    with st.chat_message("user"):
                        from sherma.streamlit.components.chat_ui import (
                            _render_content,
                        )

                        _render_content(content)

                    messages.append({"role": "user", "content": content})

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = _run_created_agent(
                                selected_agent,
                                user_input,
                                image_parts or None,
                            )
                        st.markdown(response)

                    messages.append({"role": "assistant", "content": response})

                    st.rerun()


if __name__ == "__main__":
    main()
