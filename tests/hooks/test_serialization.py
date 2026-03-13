"""Tests for hook context serialization/deserialization."""

from __future__ import annotations

from unittest.mock import MagicMock

from sherma.hooks.serialization import deserialize_into_context, serialize_context
from sherma.hooks.types import (
    AfterGraphInvokeContext,
    AfterToolCallContext,
    BeforeAgentCallContext,
    BeforeLLMCallContext,
    BeforeSkillLoadContext,
    GraphInvokeContext,
    NodeEnterContext,
    OnErrorContext,
    OnNodeErrorContext,
)


class TestSerializeContext:
    def test_fully_serializable_context(self):
        ctx = GraphInvokeContext(
            agent_id="a1", thread_id="t1", config={"key": "val"}, input={"msg": "hi"}
        )
        result = serialize_context(ctx)
        assert result == {
            "agent_id": "a1",
            "thread_id": "t1",
            "config": {"key": "val"},
            "input": {"msg": "hi"},
        }

    def test_strips_node_context(self):
        nc = MagicMock()
        ctx = NodeEnterContext(
            node_context=nc, node_name="llm1", node_type="call_llm", state={"k": "v"}
        )
        result = serialize_context(ctx)
        assert "node_context" not in result
        assert result["node_name"] == "llm1"
        assert result["node_type"] == "call_llm"
        assert result["state"] == {"k": "v"}

    def test_strips_agent_field(self):
        nc = MagicMock()
        agent = MagicMock()
        ctx = BeforeAgentCallContext(
            node_context=nc,
            node_name="agent1",
            input_value="hello",
            agent=agent,
            state={},
        )
        result = serialize_context(ctx)
        assert "node_context" not in result
        assert "agent" not in result
        assert result["input_value"] == "hello"

    def test_serializes_error_field(self):
        ctx = OnErrorContext(
            agent_id="a1",
            thread_id="t1",
            config={},
            input={},
            error=RuntimeError("boom"),
        )
        result = serialize_context(ctx)
        assert result["error"] == {"type": "RuntimeError", "message": "boom"}

    def test_serializes_none_error(self):
        ctx = OnErrorContext(
            agent_id="a1", thread_id="t1", config={}, input={}, error=None
        )
        result = serialize_context(ctx)
        assert result["error"] is None

    def test_serializes_complex_fields_as_observation(self):
        nc = MagicMock()
        ctx = BeforeLLMCallContext(
            node_context=nc,
            node_name="n",
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="be helpful",
            tools=[],
            state={"k": 1},
        )
        result = serialize_context(ctx)
        assert "node_context" not in result
        assert result["system_prompt"] == "be helpful"
        assert result["state"] == {"k": 1}
        # messages and tools are in complex fields, serialized for observation
        assert "messages" in result
        assert "tools" in result

    def test_skill_load_context_no_node_context(self):
        ctx = BeforeSkillLoadContext(node_context=None, skill_id="s1", version="1.0")
        result = serialize_context(ctx)
        assert "node_context" not in result
        assert result == {"skill_id": "s1", "version": "1.0"}


class TestDeserializeIntoContext:
    def test_round_trip_fully_serializable(self):
        original = GraphInvokeContext(
            agent_id="a1", thread_id="t1", config={"key": "val"}, input={"msg": "hi"}
        )
        data = {
            "agent_id": "a1",
            "thread_id": "t1",
            "config": {"new": True},
            "input": {"msg": "hi"},
        }
        result = deserialize_into_context(GraphInvokeContext, data, original)
        assert isinstance(result, GraphInvokeContext)
        assert result.config == {"new": True}
        assert result.agent_id == "a1"

    def test_reattaches_node_context(self):
        nc = MagicMock()
        original = NodeEnterContext(
            node_context=nc, node_name="n", node_type="call_llm", state={"old": 1}
        )
        data = {"node_name": "n", "node_type": "call_llm", "state": {"new": 2}}
        result = deserialize_into_context(NodeEnterContext, data, original)
        assert result.node_context is nc
        assert result.state == {"new": 2}

    def test_reattaches_agent(self):
        nc = MagicMock()
        agent = MagicMock()
        original = BeforeAgentCallContext(
            node_context=nc, node_name="n", input_value="old", agent=agent, state={}
        )
        data = {"node_name": "n", "input_value": "new", "state": {"updated": True}}
        result = deserialize_into_context(BeforeAgentCallContext, data, original)
        assert result.node_context is nc
        assert result.agent is agent
        assert result.input_value == "new"

    def test_reattaches_error(self):
        original_error = RuntimeError("original")
        original = OnErrorContext(
            agent_id="a1", thread_id="t1", config={}, input={}, error=original_error
        )
        data = {"agent_id": "a1", "thread_id": "t1", "config": {}, "input": {}}
        result = deserialize_into_context(OnErrorContext, data, original)
        assert result.error is original_error

    def test_keeps_complex_fields_from_original(self):
        nc = MagicMock()
        original_messages = [MagicMock()]
        original_tools = [MagicMock()]
        original = BeforeLLMCallContext(
            node_context=nc,
            node_name="n",
            messages=original_messages,
            system_prompt="old",
            tools=original_tools,
            state={"old": 1},
        )
        data = {"node_name": "n", "system_prompt": "new", "state": {"new": 2}}
        result = deserialize_into_context(BeforeLLMCallContext, data, original)
        assert result.messages is original_messages
        assert result.tools is original_tools
        assert result.system_prompt == "new"
        assert result.state == {"new": 2}

    def test_missing_field_falls_back_to_original(self):
        original = AfterGraphInvokeContext(
            agent_id="a1",
            thread_id="t1",
            config={"k": "v"},
            input={"msg": "hi"},
            result={"out": "ok"},
        )
        # Only partial data returned from remote
        data = {"result": {"out": "modified"}}
        result = deserialize_into_context(AfterGraphInvokeContext, data, original)
        assert result.agent_id == "a1"
        assert result.result == {"out": "modified"}

    def test_on_node_error_reattaches_error_and_node_context(self):
        nc = MagicMock()
        err = ValueError("test")
        original = OnNodeErrorContext(
            node_context=nc,
            node_name="n",
            node_type="call_llm",
            error=err,
            state={"k": 1},
        )
        data = {"node_name": "n", "node_type": "call_llm", "state": {"k": 2}}
        result = deserialize_into_context(OnNodeErrorContext, data, original)
        assert result.node_context is nc
        assert result.error is err
        assert result.state == {"k": 2}

    def test_after_tool_call_round_trip(self):
        nc = MagicMock()
        original = AfterToolCallContext(
            node_context=nc,
            node_name="tools",
            result={"output": "old"},
            state={"k": 1},
        )
        data = {"node_name": "tools", "result": {"output": "new"}, "state": {"k": 2}}
        result = deserialize_into_context(AfterToolCallContext, data, original)
        assert result.node_context is nc
        assert result.result == {"output": "new"}
        assert result.state == {"k": 2}
