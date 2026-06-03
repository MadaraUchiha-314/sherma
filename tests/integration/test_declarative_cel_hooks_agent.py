"""Integration test: declarative CEL hooks agent (issue #82 / #68).

Mirrors examples/declarative_cel_hooks_agent/agent.yaml with a mocked LLM.
An ``after_interrupt`` CEL hook (declared inline in the YAML) sets the
``response_type`` state field from the resumed HumanMessage's
``additional_kwargs["decision"]``, and a CEL edge routes on it — so the whole
approval flow works with no Python hook class.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from sherma.langgraph.declarative.agent import DeclarativeAgent
from sherma.langgraph.declarative.loader import RegistryBundle
from tests.integration.conftest import FakeChatModel

CEL_HOOKS_YAML_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "declarative_cel_hooks_agent"
    / "agent.yaml"
)


async def _make_agent(responses: list[AIMessage]) -> DeclarativeAgent:
    fake_model = FakeChatModel(responses=responses)
    registries = RegistryBundle(chat_models={"openai-gpt-4o-mini": fake_model})
    agent = DeclarativeAgent(
        id="cel-hooks-approval-agent",
        version="1.0.0",
        yaml_path=CEL_HOOKS_YAML_PATH,
    )
    agent._registries = registries
    return agent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cel_hook_sets_response_type_and_approves():
    """after_interrupt CEL hook sets response_type='approve' -> routes to end."""
    agent = await _make_agent(
        responses=[AIMessage(content="Draft: a crisp autumn breeze.")],
    )
    compiled = await agent.get_graph()
    config = {"configurable": {"thread_id": "cel-hooks-approve"}}

    result = await compiled.ainvoke(
        {"messages": [HumanMessage(content="Write a haiku about autumn")]},
        config,  # type: ignore[arg-type]
    )
    assert result.get("__interrupt__"), "Expected graph to pause at interrupt"

    resume_msg = HumanMessage(
        content="approve",
        additional_kwargs={"decision": "approve"},
    )
    result = await compiled.ainvoke(
        Command(resume=[resume_msg]),
        config,  # type: ignore[arg-type]
    )

    # CEL hook set response_type, CEL edge routed to __end__.
    assert not result.get("__interrupt__")
    assert result["response_type"] == "approve"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cel_hook_revise_then_approve():
    """A 'revise' decision loops back; a follow-up 'approve' exits."""
    agent = await _make_agent(
        responses=[
            AIMessage(content="Draft v1: too wordy."),
            AIMessage(content="Draft v2: concise."),
        ],
    )
    compiled = await agent.get_graph()
    config = {"configurable": {"thread_id": "cel-hooks-revise"}}

    await compiled.ainvoke(
        {"messages": [HumanMessage(content="Write a tagline")]},
        config,  # type: ignore[arg-type]
    )

    reject_msg = HumanMessage(
        content="tighten it up",
        additional_kwargs={"decision": "revise"},
    )
    result = await compiled.ainvoke(
        Command(resume=[reject_msg]),
        config,  # type: ignore[arg-type]
    )
    assert result.get("__interrupt__"), "Expected pause after revise"
    assert result["response_type"] == "revise"

    approve_msg = HumanMessage(
        content="approve",
        additional_kwargs={"decision": "approve"},
    )
    result = await compiled.ainvoke(
        Command(resume=[approve_msg]),
        config,  # type: ignore[arg-type]
    )
    assert not result.get("__interrupt__")
    assert result["response_type"] == "approve"
