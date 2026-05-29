"""Tests for first-class ``mcp_servers:`` support in declarative YAML."""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from sherma.entities.base import DEFAULT_TENANT_ID
from sherma.langgraph.declarative.loader import (
    _build_mcp_connection,
    populate_registries,
)
from sherma.langgraph.declarative.schema import DeclarativeConfig, MCPServerDef
from sherma.registry.bundle import RegistryBundle


def _fake_tool(name: str) -> StructuredTool:
    def _fn(**_: object) -> str:
        return ""

    return StructuredTool.from_function(
        func=_fn,
        name=name,
        description=f"fake mcp tool {name}",
    )


def test_http_server_requires_url() -> None:
    with pytest.raises(ValidationError, match="requires 'url'"):
        MCPServerDef(id="srv", transport="streamable_http")


def test_stdio_server_requires_command() -> None:
    with pytest.raises(ValidationError, match="requires 'command'"):
        MCPServerDef(id="srv", transport="stdio")


def test_http_server_rejects_command() -> None:
    with pytest.raises(ValidationError, match="must not set 'command'"):
        MCPServerDef(
            id="srv",
            transport="streamable_http",
            url="https://example.com",
            command="should-not-be-here",
        )


def test_stdio_server_rejects_url() -> None:
    with pytest.raises(ValidationError, match="must not set 'url'"):
        MCPServerDef(
            id="srv",
            transport="stdio",
            command="uvx",
            url="https://example.com",
        )


def test_build_connection_streamable_http() -> None:
    server = MCPServerDef(
        id="factset",
        transport="streamable_http",
        url="https://factset.example.com/mcp",
        headers={"Authorization": "Bearer abc"},
    )
    conn = _build_mcp_connection(server)
    assert conn == {
        "transport": "streamable_http",
        "url": "https://factset.example.com/mcp",
        "headers": {"Authorization": "Bearer abc"},
    }


def test_build_connection_stdio() -> None:
    server = MCPServerDef(
        id="local",
        transport="stdio",
        command="uvx",
        args=["my-mcp-server"],
        env={"DEBUG": "1"},
    )
    conn = _build_mcp_connection(server)
    assert conn == {
        "transport": "stdio",
        "command": "uvx",
        "args": ["my-mcp-server"],
        "env": {"DEBUG": "1"},
    }


@pytest.mark.asyncio
async def test_populate_registries_registers_mcp_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_connections: dict[str, dict[str, object]] = {}

    class FakeMultiServerMCPClient:
        def __init__(self, connections: dict[str, dict[str, object]]) -> None:
            captured_connections.update(connections)

        async def get_tools(
            self, *, server_name: str | None = None
        ) -> list[StructuredTool]:
            assert server_name == "factset"
            return [_fake_tool("fundamentals_query"), _fake_tool("estimates_lookup")]

    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        FakeMultiServerMCPClient,
    )

    config = DeclarativeConfig(
        manifest_version=1,
        mcp_servers=[
            MCPServerDef(
                id="factset",
                version="1.0.0",
                transport="streamable_http",
                url="https://factset.example.com/mcp",
            )
        ],
    )
    registries = RegistryBundle(tenant_id=DEFAULT_TENANT_ID)

    await populate_registries(config, registries)

    assert captured_connections["factset"]["url"] == "https://factset.example.com/mcp"

    fundamentals = await registries.tool_registry.get("fundamentals_query", "==1.0.0")
    estimates = await registries.tool_registry.get("estimates_lookup", "==1.0.0")
    assert fundamentals.id == "fundamentals_query"
    assert estimates.id == "estimates_lookup"


@pytest.mark.asyncio
async def test_populate_registries_applies_tool_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMultiServerMCPClient:
        def __init__(self, _: dict[str, object]) -> None: ...

        async def get_tools(
            self, *, server_name: str | None = None
        ) -> list[StructuredTool]:
            return [_fake_tool("query")]

    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        FakeMultiServerMCPClient,
    )

    config = DeclarativeConfig(
        manifest_version=1,
        mcp_servers=[
            MCPServerDef(
                id="factset",
                version="1.0.0",
                transport="streamable_http",
                url="https://example.com/mcp",
                tool_prefix="factset__",
            )
        ],
    )
    registries = RegistryBundle(tenant_id=DEFAULT_TENANT_ID)

    await populate_registries(config, registries)

    prefixed = await registries.tool_registry.get("factset__query", "==1.0.0")
    assert prefixed.id == "factset__query"


@pytest.mark.asyncio
async def test_hook_tools_are_excluded_from_agent_tool_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tools named ``hooks.*`` (sherma hook tools) must not pollute the
    agent's tool registry when the same MCP server also serves hooks."""

    class FakeMultiServerMCPClient:
        def __init__(self, _: dict[str, object]) -> None: ...

        async def get_tools(
            self, *, server_name: str | None = None
        ) -> list[StructuredTool]:
            return [
                _fake_tool("real_tool"),
                _fake_tool("hooks.before_llm_call"),
                _fake_tool("hooks.node_enter"),
            ]

    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        FakeMultiServerMCPClient,
    )

    config = DeclarativeConfig(
        manifest_version=1,
        mcp_servers=[
            MCPServerDef(
                id="hybrid-server",
                version="1.0.0",
                transport="streamable_http",
                url="https://example.com/mcp",
            )
        ],
    )
    registries = RegistryBundle(tenant_id=DEFAULT_TENANT_ID)

    await populate_registries(config, registries)

    # Real tool is registered.
    real = await registries.tool_registry.get("real_tool", "==1.0.0")
    assert real.id == "real_tool"

    # Hook-prefixed tools are NOT registered as agent tools.
    from sherma.exceptions import EntityNotFoundError

    for hook_tool_name in ("hooks.before_llm_call", "hooks.node_enter"):
        with pytest.raises(EntityNotFoundError):
            await registries.tool_registry.get(hook_tool_name, "==1.0.0")


@pytest.mark.asyncio
async def test_no_mcp_servers_is_noop() -> None:
    config = DeclarativeConfig(manifest_version=1)
    registries = RegistryBundle(tenant_id=DEFAULT_TENANT_ID)
    await populate_registries(config, registries)
