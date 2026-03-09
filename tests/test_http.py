import httpx
import pytest

from sherma.http import get_http_client


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
