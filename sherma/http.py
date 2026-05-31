from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Protocol

import httpx

_http_client_var: ContextVar[httpx.AsyncClient | None] = ContextVar(
    "sherma_http_client", default=None
)

HttpClientFactory = Callable[[], httpx.AsyncClient]


class McpHttpClientFactory(Protocol):
    """Factory shape expected by the MCP SDK's HTTP transports."""

    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient: ...


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


async def mcp_http_client_factory() -> McpHttpClientFactory:
    """Build an ``httpx_client_factory`` for the MCP SDK.

    The MCP SDK opens each HTTP transport inside ``async with`` and closes the
    returned ``httpx.AsyncClient``, so the factory cannot hand back the
    long-lived shared client. Instead it returns a fresh client per MCP
    connection that inherits configuration from the shared client:

    * ``event_hooks`` — copied
    * ``auth`` — inherited only if MCP did not supply its own
    * ``transport`` — reused (proxies, retries, mock transports flow through)
    * default headers — merged with MCP-supplied headers; **MCP-supplied
      headers win on conflict**
    * ``timeout`` — MCP's value wins; falls back to the shared client's
    """
    shared = await get_http_client()
    shared_transport: Any = getattr(shared, "_transport", None)

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        merged_headers = httpx.Headers(shared.headers)
        if headers:
            merged_headers.update(headers)

        kwargs: dict[str, Any] = {
            "headers": merged_headers,
            "timeout": timeout if timeout is not None else shared.timeout,
            "auth": auth if auth is not None else shared.auth,
            "event_hooks": {k: list(v) for k, v in shared.event_hooks.items()},
        }
        if shared_transport is not None:
            kwargs["transport"] = shared_transport
        return httpx.AsyncClient(**kwargs)

    return factory
