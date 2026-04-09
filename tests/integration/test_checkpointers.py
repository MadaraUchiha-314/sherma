"""Integration tests for redis/postgres checkpointers.

Skipped by default — enable by setting the corresponding
``SHERMA_TEST_REDIS_URL`` / ``SHERMA_TEST_POSTGRES_URL`` environment
variable to point at a running instance.

These tests verify that the full declarative pipeline can persist
state across two graph invocations sharing the same ``thread_id``.
"""

from __future__ import annotations

import os
import uuid

import pytest

from sherma.langgraph.declarative.agent import DeclarativeAgent

_REDIS_URL = os.environ.get("SHERMA_TEST_REDIS_URL")
_POSTGRES_URL = os.environ.get("SHERMA_TEST_POSTGRES_URL")


_REDIS_YAML_TEMPLATE = """\
manifest_version: 1

checkpointer:
  type: redis
  url: {url}

agents:
  test-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
        - name: counter
          type: int
          default: 0
    graph:
      entry_point: bump
      nodes:
        - name: bump
          type: set_state
          args:
            values:
              counter: 'state.counter + 1'
      edges: []
"""


_POSTGRES_YAML_TEMPLATE = """\
manifest_version: 1

checkpointer:
  type: postgres
  url: {url}

agents:
  test-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
        - name: counter
          type: int
          default: 0
    graph:
      entry_point: bump
      nodes:
        - name: bump
          type: set_state
          args:
            values:
              counter: 'state.counter + 1'
      edges: []
"""


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    _REDIS_URL is None,
    reason="SHERMA_TEST_REDIS_URL not set",
)
async def test_redis_checkpointer_persists_state():
    """State bumped by the first invocation survives into the second."""
    yaml = _REDIS_YAML_TEMPLATE.format(url=_REDIS_URL)
    thread_id = f"sherma-test-{uuid.uuid4()}"

    async with DeclarativeAgent(
        id="test-agent", version="1.0.0", yaml_content=yaml
    ) as agent:
        graph = await agent.get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        first = await graph.ainvoke({"messages": [], "counter": 0}, config)
        assert first["counter"] == 1

    # Fresh agent, same thread_id -> counter should have been persisted.
    async with DeclarativeAgent(
        id="test-agent", version="1.0.0", yaml_content=yaml
    ) as agent:
        graph = await agent.get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        second = await graph.ainvoke(None, config)
        assert second["counter"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    _POSTGRES_URL is None,
    reason="SHERMA_TEST_POSTGRES_URL not set",
)
async def test_postgres_checkpointer_persists_state():
    """State bumped by the first invocation survives into the second."""
    yaml = _POSTGRES_YAML_TEMPLATE.format(url=_POSTGRES_URL)
    thread_id = f"sherma-test-{uuid.uuid4()}"

    async with DeclarativeAgent(
        id="test-agent", version="1.0.0", yaml_content=yaml
    ) as agent:
        graph = await agent.get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        first = await graph.ainvoke({"messages": [], "counter": 0}, config)
        assert first["counter"] == 1

    async with DeclarativeAgent(
        id="test-agent", version="1.0.0", yaml_content=yaml
    ) as agent:
        graph = await agent.get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        second = await graph.ainvoke(None, config)
        assert second["counter"] == 1
