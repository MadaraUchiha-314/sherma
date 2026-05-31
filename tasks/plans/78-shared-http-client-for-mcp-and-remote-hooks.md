# Plan: Shared HTTP Client for MCP and Remote-Hook Executors

## Context

Sherma exposes a single shared `httpx.AsyncClient` via `sherma.http.get_http_client()`
(cached in a `ContextVar`). Today it's used by LLM calls (OpenAI/Anthropic),
A2A remote agents, and the skill/prompt/skill-card registries. Two transports
bypass it:

| Caller | Today |
|---|---|
| `RemoteHookExecutor` (`sherma/hooks/remote.py`) | Builds its own throwaway `httpx.AsyncClient(timeout=...)` in `_get_client()`. Never closed (leaks). |
| `MCPHookExecutor` (`sherma/hooks/mcp_executor.py`) | Calls `streamablehttp_client(...)` / `sse_client(...)` from the `mcp` SDK with only `url` + `headers`. Lets the MCP SDK build its own httpx client. |

This means user-configured `event_hooks`, `auth`, `transport`, base headers,
proxies, or timeouts on the shared client do **not** apply to remote hooks or
MCP hook traffic — breaking the "configure once" expectation.

## Goal

Route all HTTP traffic from `RemoteHookExecutor` and `MCPHookExecutor` through
configuration that originates from the shared `httpx.AsyncClient`, so users
who customize the shared client (headers, auth, event hooks, transport,
proxies) see those customizations applied consistently across LLM, A2A,
remote hooks, and MCP hooks.

## Design

### 1. New helper: `sherma/http.py` — `mcp_http_client_factory()`

The MCP SDK accepts an `httpx_client_factory: (headers, timeout, auth) -> httpx.AsyncClient`
and **closes** the returned client when its `async with` exits. So we cannot
hand back the literal shared client. Instead, return a factory that builds a
**fresh** `httpx.AsyncClient` per MCP connection, inheriting config from the
shared client:

- `event_hooks` — copied (so request/response interceptors apply to MCP)
- `auth` — inherited only if MCP didn't pass its own
- `transport` — reused (lets users plug in proxies, retries, mocks)
- default `headers` — merged with MCP-supplied headers; **MCP-supplied
  headers win on conflict** (server config takes precedence over global
  defaults)
- `timeout` — MCP's value wins; falls back to shared client's `timeout`

Shape:

```python
McpHttpClientFactory = Callable[
    [dict[str, str] | None, httpx.Timeout | None, httpx.Auth | None],
    httpx.AsyncClient,
]

async def mcp_http_client_factory() -> McpHttpClientFactory:
    shared = await get_http_client()
    def factory(headers=None, timeout=None, auth=None):
        merged_headers = {**dict(shared.headers), **(headers or {})}
        return httpx.AsyncClient(
            headers=merged_headers,
            timeout=timeout if timeout is not None else shared.timeout,
            auth=auth if auth is not None else shared.auth,
            transport=getattr(shared, "_transport", None),
            event_hooks=dict(shared.event_hooks),
        )
    return factory
```

Notes:
- `httpx.AsyncClient._transport` is the resolved transport; reusing it is
  safe (transports are designed to be shared).
- We construct the factory inside an async function so the shared client is
  guaranteed to exist before MCP starts using it.

### 2. `sherma/hooks/remote.py` — use the shared client directly

Replace the private client with `get_http_client()` and apply `self._timeout`
per-request:

```python
async def _call_rpc(self, method, params):
    ...
    client = await get_http_client()
    resp = await client.post(
        self._url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=self._timeout,
    )
    ...
```

- Drop `_get_client()` and `self._client` state.
- `_timeout` is preserved as a per-request override (no behavior change for
  existing users).
- Side benefit: the previous private client was never closed (leak); this
  removes the leak.

### 3. `sherma/hooks/mcp_executor.py` — pass the factory

In `connect()`, build the factory once and pass it to both transports:

```python
from sherma.http import mcp_http_client_factory
...
factory = await mcp_http_client_factory()

# streamable_http branch:
streamablehttp_client(
    url=self._transport.url,
    headers=...,
    httpx_client_factory=factory,
)

# sse branch:
sse_client(
    url=self._transport.url,
    headers=...,
    httpx_client_factory=factory,
)
```

stdio transport is unaffected — no HTTP involved.

### 4. Tests

- `tests/test_http.py`
  - New `test_mcp_http_client_factory_inherits_headers` — confirms merged
    headers, MCP-wins-on-conflict.
  - New `test_mcp_http_client_factory_inherits_event_hooks_and_transport`.
  - New `test_mcp_http_client_factory_mcp_auth_overrides`.
- `tests/hooks/test_remote.py`
  - Update mocks: assert the executor calls `sherma.hooks.remote.get_http_client`
    and uses that client for `post`.
  - Assert per-request `timeout` is passed.
- `tests/hooks/test_mcp_executor.py`
  - Assert `httpx_client_factory=` kwarg is passed into the patched
    `streamablehttp_client` and `sse_client`.

### 5. Docs and skill (per CLAUDE.md requirements)

- `docs/api-reference.md` (+ `skills/sherma/references/api-reference.md`):
  document `mcp_http_client_factory()` alongside `get_http_client()`.
- `docs/hooks.md` (+ skill copy): in the "Remote Hooks (JSON-RPC)" and
  "Remote Hooks (MCP)" sections, add a short paragraph stating that both
  transports inherit config from the shared HTTP client, and link to
  `get_http_client()`.
- `skills/sherma/SKILL.md`: add `mcp_http_client_factory` to the
  `sherma.http` API surface listing.

## Non-goals

- Per-call-type *distinct* clients (e.g., a separate "LLM client" vs
  "MCP client"). Out of scope; revisit if real demand surfaces. Users who
  want this can already differentiate inside their `event_hooks` by
  inspecting `request.url`.
- Replacing the `ContextVar` design with named/registered clients.
- stdio MCP transport changes.

## Implementation Order

1. Add `mcp_http_client_factory()` to `sherma/http.py` + tests.
2. Switch `RemoteHookExecutor` to shared client + update its tests.
3. Wire factory into `MCPHookExecutor` + update its tests.
4. Update `docs/` and mirror to `skills/sherma/references/`; update
   `skills/sherma/SKILL.md`.
5. Run `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -m "not integration"`.

## Plan Revisions

- **Header precedence** (confirmed in chat): MCP-supplied headers win over
  shared-client defaults on conflict. Shared headers are merged in as
  defaults so users can add observability/`User-Agent`/etc. globally
  without overriding explicit MCP server config.
- **Transport reuse** (confirmed in chat): the factory reuses the shared
  client's resolved `_transport`, so user-installed proxies, retries, and
  `httpx.MockTransport` in tests flow through to MCP traffic. `getattr`
  with a `None` fallback guards against future httpx renames.
- **Header merging implementation**: initial sketch used `dict.update`
  on a plain dict; this broke for case-insensitive headers like
  `Authorization`, where httpx's `AsyncClient(headers=...)` constructor
  appends rather than overwrites. Switched to building an
  `httpx.Headers` instance from the shared client's headers and calling
  `.update(headers)` on it — `httpx.Headers.update` overwrites
  case-insensitively, giving the desired "MCP wins on conflict"
  semantics.
- **`McpHttpClientFactory` typing**: initially defined as a `Callable`
  type alias, which pyright treated as positional-only — incompatible
  with the MCP SDK's keyword-arg `Protocol`. Changed to a `typing.Protocol`
  matching MCP's keyword signature so the factory is assignable to
  `streamablehttp_client(httpx_client_factory=...)` without casts.

## Risks

- **Transport reuse via private attribute (`_transport`)**: httpx doesn't
  expose this publicly. If a future httpx version renames it, the factory
  silently falls back to `None` (default transport). Mitigation: `getattr`
  with default; add a test that pins the behavior.
- **MCP SDK closing the factory-returned client**: by design we hand back a
  fresh client per connection, so this is fine.
- **Header precedence change**: previously MCP servers got *only* their
  configured headers. After this change they also see shared-client headers
  as defaults. This could leak unintended headers (e.g., an `Authorization`
  meant for the LLM). Mitigation: documented behavior; MCP-supplied headers
  always win on conflict, so explicit MCP headers are never overwritten.
