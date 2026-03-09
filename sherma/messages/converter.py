"""Bidirectional lossless conversion between A2A and LangGraph messages."""

from __future__ import annotations

import uuid

from a2a.types import DataPart, Message, Part, Role, TextPart
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

_METADATA_KEY = "a2a_metadata"


def _a2a_part_to_content_block(part: Part) -> dict[str, object] | str:
    """Convert a single A2A Part to a LangGraph content block."""
    root = part.root
    if root.kind == "text":
        return root.text
    if root.kind == "data":
        block: dict[str, object] = {"type": "data", "data": root.data}
        if root.metadata is not None:
            block["metadata"] = root.metadata
        return block
    return {"type": root.kind, "raw": root.model_dump()}


def _content_block_to_a2a_part(block: dict[str, object] | str) -> Part:
    """Convert a LangGraph content block to an A2A Part."""
    if isinstance(block, str):
        return Part(root=TextPart(text=block))
    block_type = block.get("type")
    if block_type == "text":
        return Part(root=TextPart(text=str(block.get("text", ""))))
    if block_type == "data":
        data = block.get("data")
        metadata = block.get("metadata")
        return Part(
            root=DataPart(
                data=data if isinstance(data, dict) else {},
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
    return Part(root=TextPart(text=str(block)))


def a2a_to_langgraph(message: Message) -> list[BaseMessage]:
    """Convert an A2A Message to a list of LangGraph BaseMessages.

    Preserves message_id, task_id, context_id in additional_kwargs.
    """
    parts = message.parts
    metadata: dict[str, str | None] = {}
    if message.message_id:
        metadata["messageId"] = message.message_id
    if message.task_id:
        metadata["taskId"] = message.task_id
    if message.context_id:
        metadata["contextId"] = message.context_id

    # Build content
    if len(parts) == 1 and parts[0].root.kind == "text":
        content: str | list[dict[str, object] | str] = parts[0].root.text
    else:
        content = [_a2a_part_to_content_block(p) for p in parts]

    additional_kwargs: dict[str, object] = {}
    if metadata:
        additional_kwargs[_METADATA_KEY] = metadata

    if message.role == Role.user:
        return [HumanMessage(content=content, additional_kwargs=additional_kwargs)]
    return [AIMessage(content=content, additional_kwargs=additional_kwargs)]


def langgraph_to_a2a(message: BaseMessage) -> Message:
    """Convert a LangGraph BaseMessage to an A2A Message.

    Restores message_id, task_id, context_id from additional_kwargs.
    """
    role = Role.user if message.type == "human" else Role.agent
    content = message.content
    additional_kwargs = message.additional_kwargs

    # Build parts
    if isinstance(content, str):
        parts = [Part(root=TextPart(text=content))]
    elif isinstance(content, list):
        parts = [_content_block_to_a2a_part(block) for block in content]
    else:
        parts = [Part(root=TextPart(text=str(content)))]

    metadata = additional_kwargs.get(_METADATA_KEY, {})

    return Message(
        message_id=metadata.get("messageId", str(uuid.uuid4())),
        role=role,
        parts=parts,
        task_id=metadata.get("taskId"),
        context_id=metadata.get("contextId"),
    )
