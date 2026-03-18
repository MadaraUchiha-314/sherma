"""@tool functions for the Agent Designer agent.

These tools let the LLM save/validate generated files. They communicate
with the Streamlit session via the ContextVar-based store in
``sherma.streamlit.storage``.
"""

from __future__ import annotations

import json

import yaml
from langchain_core.tools import tool

from sherma.streamlit.storage import get_store


@tool
def save_agent_yaml(yaml_content: str, filename: str = "agent.yaml") -> str:
    """Save a declarative agent YAML config file.

    Validates the YAML structure before saving. Returns success or
    validation error details.
    """
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        return f"Invalid YAML syntax: {exc}"

    if not isinstance(data, dict):
        return "Error: YAML root must be a mapping"

    try:
        from sherma.langgraph.declarative.loader import load_declarative_config

        load_declarative_config(yaml_content=yaml_content)
    except Exception as exc:
        return f"Validation warning (saved anyway): {exc}"
    finally:
        store = get_store()
        store[filename] = yaml_content

    return f"Saved and validated '{filename}' ({len(yaml_content)} bytes)"


@tool
def save_tool_code(filename: str, code: str) -> str:
    """Save a Python tool file to the generated files store."""
    store = get_store()
    store[filename] = code
    return f"Saved tool file '{filename}' ({len(code)} bytes)"


@tool
def save_prompt_file(filename: str, content: str) -> str:
    """Save a prompt or text file to the generated files store."""
    store = get_store()
    store[filename] = content
    return f"Saved prompt file '{filename}' ({len(content)} bytes)"


@tool
def save_file(filename: str, content: str) -> str:
    """Save a general-purpose file to the generated files store."""
    store = get_store()
    store[filename] = content
    return f"Saved file '{filename}' ({len(content)} bytes)"


@tool
def get_generated_files() -> str:
    """List all generated files with their sizes and first few lines."""
    store = get_store()
    if not store:
        return "No files generated yet."
    result = []
    for name, content in store.items():
        lines = content.splitlines()
        preview = "\n".join(lines[:5])
        if len(lines) > 5:
            preview += "\n..."
        result.append({"filename": name, "size": len(content), "preview": preview})
    return json.dumps(result, indent=2)


@tool
def validate_agent_yaml(yaml_content: str) -> str:
    """Validate a YAML string against the sherma declarative agent schema.

    Returns 'valid' or detailed error information.
    """
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        return f"Invalid YAML syntax: {exc}"

    if not isinstance(data, dict):
        return "Error: YAML root must be a mapping"

    try:
        from sherma.langgraph.declarative.loader import load_declarative_config

        load_declarative_config(yaml_content=yaml_content)
    except Exception as exc:
        return f"Validation error: {exc}"

    return "valid"
