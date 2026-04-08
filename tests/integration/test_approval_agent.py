"""Integration test: approval agent with structured interrupt resume.

Mirrors examples/approval_agent/main_structured_resume.py with a mocked LLM.
Drives the graph directly (as that example does) and passes a structured
HumanMessage into ``Command(resume=...)``. The CEL edge in the example's
``agent.yaml`` routes on ``additional_kwargs["decision"]``, so if the
interrupt node stringified the resume value, the approval path would never
be taken.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from sherma.langgraph.declarative.agent import DeclarativeAgent
from sherma.langgraph.declarative.loader import RegistryBundle
from tests.integration.conftest import FakeChatModel

APPROVAL_YAML_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "approval_agent"
    / "agent.yaml"
)


def _make_agent(responses: list[AIMessage]) -> DeclarativeAgent:
    fake_model = FakeChatModel(responses=responses)
    # The example YAML references the `openai-gpt-4o-mini` LLM id; we
    # substitute the fake chat model via the registry bundle.
    registries = RegistryBundle(chat_models={"openai-gpt-4o-mini": fake_model})
    # Load the real example YAML so this test doubles as a smoke check
    # that examples/approval_agent/agent.yaml still parses and compiles.
    agent = DeclarativeAgent(
        id="approval-agent",
        version="1.0.0",
        yaml_path=APPROVAL_YAML_PATH,
    )
    agent._registries = registries
    return agent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_approval_agent_structured_resume_approves_without_hook():
    """Client-supplied additional_kwargs routes to __end__ with no hook."""
    agent = _make_agent(
        responses=[
            AIMessage(content="Draft: a crisp autumn breeze."),
        ],
    )
    compiled = await agent.get_graph()
    config = {"configurable": {"thread_id": "approval-structured-approve"}}

    # Initial invoke — LLM drafts and graph pauses at get_approval.
    result = await compiled.ainvoke(
        {"messages": [HumanMessage(content="Write a haiku about autumn")]},
        config,  # type: ignore[arg-type]
    )
    assert result.get("__interrupt__"), "Expected graph to pause at interrupt"

    # Structured resume — no ApprovalTaggingHook in sight.
    resume_msg = HumanMessage(
        content="approve",
        additional_kwargs={"decision": "approve"},
    )
    result = await compiled.ainvoke(
        Command(resume=[resume_msg]),
        config,  # type: ignore[arg-type]
    )

    # No interrupt → CEL routed to __end__.
    assert not result.get("__interrupt__")

    # The resume HumanMessage is preserved verbatim (metadata intact).
    human_msgs = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    tagged = [m for m in human_msgs if m.additional_kwargs.get("decision") == "approve"]
    assert len(tagged) == 1
    assert tagged[0].content == "approve"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_approval_agent_structured_resume_revises_then_approves():
    """A 'revise' decision loops back; a follow-up 'approve' exits."""
    agent = _make_agent(
        responses=[
            AIMessage(content="Draft v1: too wordy."),
            AIMessage(content="Draft v2: concise."),
        ],
    )
    compiled = await agent.get_graph()
    config = {"configurable": {"thread_id": "approval-structured-revise"}}

    # Draft v1 → pause.
    await compiled.ainvoke(
        {"messages": [HumanMessage(content="Write a tagline")]},
        config,  # type: ignore[arg-type]
    )

    # Reject v1 with structured metadata → revise node runs → pause again.
    reject_msg = HumanMessage(
        content="tighten it up",
        additional_kwargs={"decision": "revise"},
    )
    result = await compiled.ainvoke(
        Command(resume=[reject_msg]),
        config,  # type: ignore[arg-type]
    )
    assert result.get("__interrupt__"), "Expected pause after revise"

    # Approve v2 with structured metadata → exit.
    approve_msg = HumanMessage(
        content="approve",
        additional_kwargs={"decision": "approve"},
    )
    result = await compiled.ainvoke(
        Command(resume=[approve_msg]),
        config,  # type: ignore[arg-type]
    )
    assert not result.get("__interrupt__")

    # Both structured resume messages are preserved in state.
    tagged = [
        m
        for m in result["messages"]
        if isinstance(m, HumanMessage) and m.additional_kwargs.get("decision")
    ]
    decisions = [m.additional_kwargs["decision"] for m in tagged]
    assert decisions == ["revise", "approve"]
