import httpx
import pytest

from sherma.http import get_http_client, mcp_http_client_factory


@pytest.mark.asyncio
async def test_get_http_client_default():
    client = await get_http_client()
    assert isinstance(client, httpx.AsyncClient)
    await client.aclose()


@pytest.mark.asyncio
async def test_get_http_client_with_instance():
    custom = httpx.AsyncClient()
    try:
        client = await get_http_client(custom)
        assert client is custom
    finally:
        await custom.aclose()


@pytest.mark.asyncio
async def test_get_http_client_with_factory():
    custom = httpx.AsyncClient()

    def factory() -> httpx.AsyncClient:
        return custom

    try:
        client = await get_http_client(factory)
        assert client is custom
    finally:
        await custom.aclose()


@pytest.mark.asyncio
async def test_mcp_factory_inherits_headers_and_mcp_wins_on_conflict():
    """Shared headers are merged in; MCP-supplied headers win on conflict."""
    shared = httpx.AsyncClient(
        headers={"X-Trace": "abc", "Authorization": "Bearer shared"}
    )
    try:
        await get_http_client(shared)
        factory = await mcp_http_client_factory()
        c = factory({"Authorization": "Bearer mcp"}, None, None)
        try:
            assert c.headers.get("x-trace") == "abc"
            assert c.headers.get("authorization") == "Bearer mcp"
        finally:
            await c.aclose()
    finally:
        await shared.aclose()


@pytest.mark.asyncio
async def test_mcp_factory_inherits_event_hooks_and_transport():
    """event_hooks and transport flow through to MCP-created clients."""

    async def hook(_request):
        return None

    transport = httpx.MockTransport(lambda req: httpx.Response(204))
    shared = httpx.AsyncClient(transport=transport, event_hooks={"request": [hook]})
    try:
        await get_http_client(shared)
        factory = await mcp_http_client_factory()
        c = factory(None, None, None)
        try:
            assert hook in c.event_hooks["request"]
            assert c._transport is transport  # type: ignore[attr-defined]
        finally:
            await c.aclose()
    finally:
        await shared.aclose()


@pytest.mark.asyncio
async def test_mcp_factory_mcp_auth_and_timeout_override_shared():
    """MCP-supplied auth/timeout take precedence over the shared client's."""
    shared_auth = httpx.BasicAuth("u", "p")
    shared = httpx.AsyncClient(auth=shared_auth, timeout=httpx.Timeout(10.0))
    try:
        await get_http_client(shared)
        factory = await mcp_http_client_factory()

        # No MCP-supplied auth: inherit shared
        c1 = factory(None, None, None)
        try:
            assert c1.auth is shared_auth
        finally:
            await c1.aclose()

        # MCP supplies its own auth: it wins
        mcp_auth = httpx.BasicAuth("x", "y")
        c2 = factory(None, httpx.Timeout(2.0), mcp_auth)
        try:
            assert c2.auth is mcp_auth
            assert c2.timeout == httpx.Timeout(2.0)
        finally:
            await c2.aclose()
    finally:
        await shared.aclose()
