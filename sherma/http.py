from collections.abc import Callable
from contextvars import ContextVar

import httpx

_http_client_var: ContextVar[httpx.AsyncClient | None] = ContextVar(
    "sherma_http_client", default=None
)

HttpClientFactory = Callable[[], httpx.AsyncClient]


async def get_http_client(
    client: httpx.AsyncClient | HttpClientFactory | None = None,
) -> httpx.AsyncClient:
    """Get or create an httpx AsyncClient.

    If a client instance is provided, it is used directly and cached.
    If a factory is provided, it is called to create a client and cached.
    If None, a default client is created and cached in a ContextVar.
    """
    if client is not None:
        if isinstance(client, httpx.AsyncClient):
            _http_client_var.set(client)
            return client
        instance = client()
        _http_client_var.set(instance)
        return instance

    existing = _http_client_var.get()
    if existing is not None:
        return existing

    new_client = httpx.AsyncClient()
    _http_client_var.set(new_client)
    return new_client
