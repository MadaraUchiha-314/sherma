"""Tests for declarative on_error: retry, fallback, and interrupt safety."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from sherma.exceptions import DeclarativeConfigError
from sherma.langgraph.declarative.cel_engine import CelEngine
from sherma.langgraph.declarative.edges import build_conditional_router
from sherma.langgraph.declarative.loader import load_declarative_config, validate_config
from sherma.langgraph.declarative.nodes import (
    INTERNAL_STATE_KEY,
    NodeContext,
    _compute_delay,
    _store_error_and_fallback,
    build_call_llm_node,
    build_tool_node,
)
from sherma.langgraph.declarative.schema import (
    BranchDef,
    CallLLMArgs,
    DeclarativeConfig,
    EdgeDef,
    NodeDef,
    OnErrorDef,
    PromptMessageDef,
    RegistryRef,
    RetryPolicy,
    ToolNodeArgs,
)
from sherma.langgraph.declarative.transform import (
    HAS_ERROR_FALLBACK,
    inject_fallback_edges,
)


def _make_ctx(node_def: NodeDef, hook_manager: object | None = None) -> NodeContext:
    config = DeclarativeConfig(manifest_version=1)
    return NodeContext(config=config, node_def=node_def, hook_manager=hook_manager)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------


class TestRetryPolicyDefaults:
    def test_defaults(self):
        r = RetryPolicy()
        assert r.max_attempts == 3
        assert r.strategy == "exponential"
        assert r.delay == 1.0
        assert r.max_delay == 30.0

    def test_fixed_strategy(self):
        r = RetryPolicy(strategy="fixed", delay=2.0, max_attempts=5)
        assert r.strategy == "fixed"
        assert r.max_attempts == 5


class TestOnErrorDef:
    def test_retry_only(self):
        o = OnErrorDef(retry=RetryPolicy(max_attempts=2))
        assert o.retry is not None
        assert o.fallback is None

    def test_fallback_only(self):
        o = OnErrorDef(fallback="error_handler")
        assert o.retry is None
        assert o.fallback == "error_handler"

    def test_both(self):
        o = OnErrorDef(retry=RetryPolicy(), fallback="handler")
        assert o.retry is not None
        assert o.fallback == "handler"


class TestNodeDefOnError:
    def test_node_def_with_on_error(self):
        n = NodeDef(
            name="agent",
            type="call_llm",
            args=CallLLMArgs(
                llm=RegistryRef(id="gpt-4"),
                prompt=[PromptMessageDef(role="system", content='"hi"')],
                state_updates={"messages": "[llm_response]"},
            ),
            on_error=OnErrorDef(
                retry=RetryPolicy(max_attempts=3),
                fallback="handler",
            ),
        )
        assert n.on_error is not None
        assert n.on_error.retry is not None
        assert n.on_error.fallback == "handler"

    def test_node_def_without_on_error(self):
        n = NodeDef(
            name="agent",
            type="call_llm",
            args=CallLLMArgs(
                llm=RegistryRef(id="gpt-4"),
                prompt=[PromptMessageDef(role="system", content='"hi"')],
                state_updates={"messages": "[llm_response]"},
            ),
        )
        assert n.on_error is None


# ---------------------------------------------------------------------------
# _compute_delay
# ---------------------------------------------------------------------------


class TestComputeDelay:
    def test_fixed_strategy(self):
        r = RetryPolicy(strategy="fixed", delay=2.0, max_delay=10.0)
        assert _compute_delay(r, 1) == 2.0
        assert _compute_delay(r, 5) == 2.0

    def test_fixed_capped(self):
        r = RetryPolicy(strategy="fixed", delay=15.0, max_delay=10.0)
        assert _compute_delay(r, 1) == 10.0

    def test_exponential_strategy(self):
        r = RetryPolicy(strategy="exponential", delay=1.0, max_delay=30.0)
        assert _compute_delay(r, 1) == 1.0  # 1 * 2^0
        assert _compute_delay(r, 2) == 2.0  # 1 * 2^1
        assert _compute_delay(r, 3) == 4.0  # 1 * 2^2
        assert _compute_delay(r, 4) == 8.0  # 1 * 2^3

    def test_exponential_capped(self):
        r = RetryPolicy(strategy="exponential", delay=1.0, max_delay=5.0)
        assert _compute_delay(r, 4) == 5.0  # 8 capped to 5


# ---------------------------------------------------------------------------
# _store_error_and_fallback
# ---------------------------------------------------------------------------


class TestStoreErrorAndFallback:
    def test_stores_error_info(self):
        state: dict = {"messages": []}
        exc = ValueError("boom")
        result = _store_error_and_fallback(state, "my_node", exc, 3, "handler")

        internal = result[INTERNAL_STATE_KEY]
        assert internal["last_error"]["node"] == "my_node"
        assert internal["last_error"]["type"] == "ValueError"
        assert internal["last_error"]["message"] == "boom"
        assert internal["last_error"]["attempt"] == 3
        assert internal["error_fallback"] == "handler"


# ---------------------------------------------------------------------------
# Interrupt safety: GraphBubbleUp re-raised
# ---------------------------------------------------------------------------


class TestInterruptSafety:
    @pytest.mark.asyncio
    async def test_graph_bubble_up_reraised_in_call_llm(self):
        from langgraph.errors import GraphBubbleUp

        node_def = NodeDef(
            name="agent",
            type="call_llm",
            args=CallLLMArgs(
                llm=RegistryRef(id="gpt-4"),
                prompt=[
                    PromptMessageDef(role="system", content='"hi"'),
                    PromptMessageDef(role="messages", content="state.messages"),
                ],
                state_updates={"messages": "[llm_response]"},
            ),
            on_error=OnErrorDef(
                retry=RetryPolicy(max_attempts=3),
                fallback="handler",
            ),
        )
        chat_model = AsyncMock()
        chat_model.ainvoke = AsyncMock(side_effect=GraphBubbleUp())
        cel = CelEngine()

        fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
        with pytest.raises(GraphBubbleUp):
            await fn({"messages": []})

        # Should NOT retry on GraphBubbleUp — only 1 call
        assert chat_model.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_graph_bubble_up_reraised_in_tool_node(self):
        from langgraph.errors import GraphBubbleUp

        from sherma.registry.tool import ToolRegistry

        node_def = NodeDef(
            name="tools",
            type="tool_node",
            args=ToolNodeArgs(),
            on_error=OnErrorDef(fallback="handler"),
        )
        tool_registry = ToolRegistry()

        ctx = _make_ctx(node_def)
        fn = build_tool_node(ctx, tool_registry=tool_registry)

        with pytest.raises(GraphBubbleUp):
            # Patch _resolve_all_registry_tools to raise GraphBubbleUp
            with patch(
                "sherma.langgraph.declarative.nodes._resolve_all_registry_tools",
                side_effect=GraphBubbleUp(),
            ):
                await fn({"messages": [AIMessage(content="hi")]})


# ---------------------------------------------------------------------------
# Retry in call_llm
# ---------------------------------------------------------------------------


class TestCallLLMRetry:
    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        node_def = NodeDef(
            name="agent",
            type="call_llm",
            args=CallLLMArgs(
                llm=RegistryRef(id="gpt-4"),
                prompt=[
                    PromptMessageDef(role="system", content='"hi"'),
                    PromptMessageDef(role="messages", content="state.messages"),
                ],
                state_updates={"messages": "[llm_response]"},
            ),
            on_error=OnErrorDef(
                retry=RetryPolicy(max_attempts=3, delay=0.01, strategy="fixed"),
            ),
        )
        chat_model = AsyncMock()
        chat_model.ainvoke = AsyncMock(
            side_effect=[
                RuntimeError("fail 1"),
                RuntimeError("fail 2"),
                AIMessage(content="success"),
            ]
        )
        cel = CelEngine()

        fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
        result = await fn({"messages": []})

        assert result["messages"][0]["content"] == "success"
        assert chat_model.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_exhausted_no_fallback_raises(self):
        node_def = NodeDef(
            name="agent",
            type="call_llm",
            args=CallLLMArgs(
                llm=RegistryRef(id="gpt-4"),
                prompt=[
                    PromptMessageDef(role="system", content='"hi"'),
                    PromptMessageDef(role="messages", content="state.messages"),
                ],
                state_updates={"messages": "[llm_response]"},
            ),
            on_error=OnErrorDef(
                retry=RetryPolicy(max_attempts=2, delay=0.01, strategy="fixed"),
            ),
        )
        chat_model = AsyncMock()
        chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("always fails"))
        cel = CelEngine()

        fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
        with pytest.raises(RuntimeError, match="always fails"):
            await fn({"messages": []})

        assert chat_model.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_exhausted_with_fallback_returns_sentinel(self):
        node_def = NodeDef(
            name="agent",
            type="call_llm",
            args=CallLLMArgs(
                llm=RegistryRef(id="gpt-4"),
                prompt=[
                    PromptMessageDef(role="system", content='"hi"'),
                    PromptMessageDef(role="messages", content="state.messages"),
                ],
                state_updates={"messages": "[llm_response]"},
            ),
            on_error=OnErrorDef(
                retry=RetryPolicy(max_attempts=2, delay=0.01, strategy="fixed"),
                fallback="error_handler",
            ),
        )
        chat_model = AsyncMock()
        chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("always fails"))
        cel = CelEngine()

        fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
        result = await fn({"messages": []})

        internal = result[INTERNAL_STATE_KEY]
        assert internal["error_fallback"] == "error_handler"
        assert internal["last_error"]["node"] == "agent"
        assert internal["last_error"]["type"] == "RuntimeError"
        assert internal["last_error"]["attempt"] == 2

    @pytest.mark.asyncio
    async def test_no_retry_when_not_configured(self):
        node_def = NodeDef(
            name="agent",
            type="call_llm",
            args=CallLLMArgs(
                llm=RegistryRef(id="gpt-4"),
                prompt=[
                    PromptMessageDef(role="system", content='"hi"'),
                    PromptMessageDef(role="messages", content="state.messages"),
                ],
                state_updates={"messages": "[llm_response]"},
            ),
        )
        chat_model = AsyncMock()
        chat_model.ainvoke = AsyncMock(side_effect=RuntimeError("fail"))
        cel = CelEngine()

        fn = build_call_llm_node(_make_ctx(node_def), chat_model, cel)
        with pytest.raises(RuntimeError, match="fail"):
            await fn({"messages": []})

        # Only 1 attempt — no retry
        assert chat_model.ainvoke.call_count == 1


# ---------------------------------------------------------------------------
# Fallback for tool_node (no retry)
# ---------------------------------------------------------------------------


class TestToolNodeFallback:
    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        from sherma.registry.tool import ToolRegistry

        node_def = NodeDef(
            name="tools",
            type="tool_node",
            args=ToolNodeArgs(),
            on_error=OnErrorDef(fallback="error_handler"),
        )
        tool_registry = ToolRegistry()
        ctx = _make_ctx(node_def)
        fn = build_tool_node(ctx, tool_registry=tool_registry)

        with patch(
            "sherma.langgraph.declarative.nodes._resolve_all_registry_tools",
            side_effect=RuntimeError("tool boom"),
        ):
            result = await fn({"messages": [AIMessage(content="hi")]})

        internal = result[INTERNAL_STATE_KEY]
        assert internal["error_fallback"] == "error_handler"
        assert internal["last_error"]["type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# inject_fallback_edges transform
# ---------------------------------------------------------------------------


def _base_yaml(nodes_yaml: str, edges_yaml: str = "edges: []") -> str:
    return f"""\
manifest_version: 1

llms:
  - id: gpt-4
    version: "1.0.0"
    model_name: gpt-4

agents:
  test-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
    graph:
      entry_point: agent
      nodes:
{nodes_yaml}
      {edges_yaml}
"""


class TestInjectFallbackEdges:
    def test_simple_edge_replaced_with_conditional(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
          on_error:
            fallback: handler
        - name: handler
          type: data_transform
          args:
            expression: '{"messages": state.messages}'""",
            """\
edges:
        - source: agent
          target: handler""",
        )
        config = load_declarative_config(yaml_content=yaml)
        result = inject_fallback_edges(config)

        agent = result.agents["test-agent"]
        # Simple edge should be replaced with conditional
        cond_edges = [
            e for e in agent.graph.edges if e.source == "agent" and e.branches
        ]
        assert len(cond_edges) == 1
        edge = cond_edges[0]
        assert edge.branches[0].condition == HAS_ERROR_FALLBACK
        assert edge.branches[0].target == "handler"
        assert edge.default == "handler"

    def test_existing_conditional_gets_fallback_prepended(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
          on_error:
            fallback: handler
        - name: handler
          type: data_transform
          args:
            expression: '{"messages": state.messages}'
        - name: other
          type: set_state
          args:
            values:
              x: '"done"'""",
            """\
edges:
        - source: agent
          branches:
            - condition: 'state.x == "go"'
              target: other
          default: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        result = inject_fallback_edges(config)

        agent = result.agents["test-agent"]
        cond_edge = next(e for e in agent.graph.edges if e.source == "agent")
        # Fallback branch should be first
        assert cond_edge.branches[0].condition == HAS_ERROR_FALLBACK
        assert cond_edge.branches[0].target == "handler"
        # Original branch preserved
        assert cond_edge.branches[1].condition == 'state.x == "go"'

    def test_no_on_error_no_changes(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        result = inject_fallback_edges(config)

        agent = result.agents["test-agent"]
        # Should still be a simple edge
        assert agent.graph.edges[0].target == "__end__"
        assert agent.graph.edges[0].branches is None

    def test_invalid_fallback_target_raises(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
          on_error:
            fallback: nonexistent""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        with pytest.raises(
            DeclarativeConfigError,
            match=r"nonexistent.*does not exist",
        ):
            inject_fallback_edges(config)


# ---------------------------------------------------------------------------
# has_error_fallback built-in condition
# ---------------------------------------------------------------------------


class TestHasErrorFallbackCondition:
    def test_routes_to_fallback_when_sentinel_set(self):
        edge = EdgeDef(
            source="agent",
            branches=[
                BranchDef(condition=HAS_ERROR_FALLBACK, target="handler"),
            ],
            default="next",
        )
        cel = CelEngine()
        router, _path_map = build_conditional_router(edge, cel)

        state = {INTERNAL_STATE_KEY: {"error_fallback": "handler"}}
        assert router(state) == "handler"

    def test_routes_to_default_when_no_sentinel(self):
        edge = EdgeDef(
            source="agent",
            branches=[
                BranchDef(condition=HAS_ERROR_FALLBACK, target="handler"),
            ],
            default="next",
        )
        cel = CelEngine()
        router, _path_map = build_conditional_router(edge, cel)

        state: dict = {}
        assert router(state) == "next"

    def test_routes_to_default_when_empty_internal(self):
        edge = EdgeDef(
            source="agent",
            branches=[
                BranchDef(condition=HAS_ERROR_FALLBACK, target="handler"),
            ],
            default="next",
        )
        cel = CelEngine()
        router, _path_map = build_conditional_router(edge, cel)

        state = {INTERNAL_STATE_KEY: {}}
        assert router(state) == "next"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestOnErrorValidation:
    def test_retry_rejected_on_tool_node(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
          args: {}
          on_error:
            retry:
              max_attempts: 3
            fallback: agent""",
            """\
edges:
        - source: agent
          target: tools
        - source: tools
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        with pytest.raises(
            DeclarativeConfigError,
            match=r"retry.*only supported on call_llm",
        ):
            validate_config(config, "test-agent")

    def test_on_error_rejected_on_data_transform(self):
        yaml = _base_yaml(
            """\
        - name: transform
          type: data_transform
          args:
            expression: '{"x": 1}'
          on_error:
            fallback: agent
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'""",
            """\
edges:
        - source: transform
          target: agent
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        with pytest.raises(
            DeclarativeConfigError,
            match=r"not supported on.*data_transform",
        ):
            validate_config(config, "test-agent")

    def test_on_error_rejected_on_set_state(self):
        yaml = _base_yaml(
            """\
        - name: init
          type: set_state
          args:
            values:
              x: '"hi"'
          on_error:
            fallback: agent
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'""",
            """\
edges:
        - source: init
          target: agent
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        with pytest.raises(
            DeclarativeConfigError,
            match=r"not supported on.*set_state",
        ):
            validate_config(config, "test-agent")

    def test_on_error_rejected_on_interrupt(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
        - name: pause
          type: interrupt
          args:
            value: '"waiting"'
          on_error:
            fallback: agent""",
            """\
edges:
        - source: agent
          target: pause
        - source: pause
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        with pytest.raises(
            DeclarativeConfigError,
            match=r"not supported on.*interrupt",
        ):
            validate_config(config, "test-agent")

    def test_invalid_retry_max_attempts(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
          on_error:
            retry:
              max_attempts: 0""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        with pytest.raises(DeclarativeConfigError, match="max_attempts must be >= 1"):
            validate_config(config, "test-agent")

    def test_invalid_retry_delay_negative(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
          on_error:
            retry:
              delay: -1""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        with pytest.raises(DeclarativeConfigError, match="delay must be >= 0"):
            validate_config(config, "test-agent")

    def test_invalid_max_delay_less_than_delay(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
          on_error:
            retry:
              delay: 5
              max_delay: 2""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        with pytest.raises(
            DeclarativeConfigError,
            match=r"max_delay.*must be >= delay",
        ):
            validate_config(config, "test-agent")

    def test_fallback_target_not_in_graph(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
          on_error:
            fallback: nonexistent""",
            """\
edges:
        - source: agent
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        with pytest.raises(
            DeclarativeConfigError,
            match=r"nonexistent.*does not exist",
        ):
            validate_config(config, "test-agent")

    def test_valid_on_error_passes(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
          on_error:
            retry:
              max_attempts: 3
              delay: 1.0
              max_delay: 10.0
            fallback: handler
        - name: handler
          type: data_transform
          args:
            expression: '{"messages": state.messages}'""",
            """\
edges:
        - source: agent
          target: handler
        - source: handler
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        # Should not raise
        validate_config(config, "test-agent")

    def test_tool_node_fallback_only_passes(self):
        yaml = _base_yaml(
            """\
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hello"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
        - name: tools
          type: tool_node
          args: {}
          on_error:
            fallback: agent""",
            """\
edges:
        - source: agent
          target: tools
        - source: tools
          target: __end__""",
        )
        config = load_declarative_config(yaml_content=yaml)
        # Should not raise — fallback without retry is valid for tool_node
        validate_config(config, "test-agent")


# ---------------------------------------------------------------------------
# YAML parsing with on_error
# ---------------------------------------------------------------------------


class TestYAMLParsing:
    def test_on_error_parsed_from_yaml(self):
        yaml = """\
manifest_version: 1

llms:
  - id: gpt-4
    model_name: gpt-4

agents:
  test-agent:
    state:
      fields:
        - name: messages
          type: list
    graph:
      entry_point: agent
      nodes:
        - name: agent
          type: call_llm
          args:
            llm: { id: gpt-4 }
            prompt:
              - role: system
                content: '"hi"'
              - role: messages
                content: 'state.messages'
            state_updates:
              messages: '[llm_response]'
          on_error:
            retry:
              max_attempts: 5
              strategy: fixed
              delay: 2.0
              max_delay: 10.0
            fallback: handler
        - name: handler
          type: data_transform
          args:
            expression: '{"messages": state.messages}'
      edges:
        - source: agent
          target: handler
        - source: handler
          target: __end__
"""
        config = load_declarative_config(yaml_content=yaml)
        node = config.agents["test-agent"].graph.nodes[0]
        assert node.on_error is not None
        assert node.on_error.retry is not None
        assert node.on_error.retry.max_attempts == 5
        assert node.on_error.retry.strategy == "fixed"
        assert node.on_error.retry.delay == 2.0
        assert node.on_error.retry.max_delay == 10.0
        assert node.on_error.fallback == "handler"
