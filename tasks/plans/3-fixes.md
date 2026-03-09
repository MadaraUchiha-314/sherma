# Plan: Align Sherma with A2A Protocol Types

## Context

The sherma agent framework currently uses loose `Any` types for its agent and executor interfaces. The A2A Python SDK (`a2a-sdk`) defines concrete interfaces — `Client` (abstract base for clients) and `AgentExecutor` (abstract base for server-side executors). We need to align sherma's interfaces with these A2A types so that:
- `Agent.send_message()` / `cancel_task()` signatures match the A2A Client methods
- `ShermaAgentExecutor` inherits from `a2a.server.agent_execution.AgentExecutor`
- Proper A2A types (`Message`, `Task`, `TaskIdParams`, `RequestContext`, `EventQueue`) are used
- Iterator return type includes all A2A event types: `Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent`
- All dependencies are **required**, not optional — this library is opinionated about its toolchain

## A2A Reference Signatures

**A2A Client (`a2a.client.client.Client`) — the proper client (NOT the deprecated `A2AClient`):**
```python
# Abstract base class. Create instances via ClientFactory.connect()
from a2a.client import Client, ClientConfig, ClientFactory

# Factory — connect from URL or AgentCard
client = await ClientFactory.connect(url_or_agent_card, client_config=ClientConfig(httpx_client=...))

# send_message — streaming, returns async iterator of ClientEvent | Message
# ClientEvent = tuple[Task, UpdateEvent] where UpdateEvent = TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None
async def send_message(
    self, request: Message, *,
    context: ClientCallContext | None = None,
    request_metadata: dict[str, Any] | None = None,
    extensions: list[str] | None = None,
) -> AsyncIterator[ClientEvent | Message]

# cancel_task — returns Task directly
async def cancel_task(
    self, request: TaskIdParams, *,
    context: ClientCallContext | None = None,
    extensions: list[str] | None = None,
) -> Task

# get_card — returns AgentCard
async def get_card(self) -> AgentCard
```

**A2A AgentExecutor:**
```python
async def execute(self, context: RequestContext, event_queue: EventQueue) -> None
async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None
```

**RequestContext key properties:**
- `message -> Message | None` (from `self._params.message` where `_params` is `MessageSendParams | None`)
- `task_id -> str | None`
- Constructor: `RequestContext(request: MessageSendParams | None, task_id: str | None, ...)`

**EventQueue:**
- `Event = Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent`
- `async enqueue_event(event: Event) -> None`
- `async dequeue_event(no_wait: bool = False) -> Event`

## Changes

### 1. `pyproject.toml` — Make all dependencies required

- Move `a2a-sdk`, `langgraph`, `langchain-core`, `langfuse`, `langchain`, `mcp` from optional to required dependencies
- Only `langchain-openai` remains optional (under `examples` extra)
- `langchain` is required by `langfuse.langchain.CallbackHandler`
- Remove the `all` extra and individual extras (`a2a`, `langgraph`, `langfuse`, `mcp`)

### 2. `sherma/entities/agent/base.py` — Update Agent interface

- All A2A types imported directly at runtime (no `TYPE_CHECKING` guards)
- Import `ClientEvent` from `a2a.client.client` and `ClientCallContext` from `a2a.client.middleware`
- `send_message` signature matches `Client.send_message` exactly:
  ```python
  def send_message(self, request: Message, *, context: ClientCallContext | None = None,
                   request_metadata: dict[str, Any] | None = None,
                   extensions: list[str] | None = None) -> AsyncIterator[ClientEvent | Message]
  ```
  **Not `async def`** — so subclasses can implement as async generators
- `cancel_task` signature matches `Client.cancel_task` exactly:
  ```python
  async def cancel_task(self, request: TaskIdParams, *, context: ClientCallContext | None = None,
                        extensions: list[str] | None = None) -> Task
  ```
- `get_card(self) -> AgentCard | None`
- `agent_card: AgentCard | None = None` — proper type, no `Any`

### 3. `sherma/a2a/executor.py` — Inherit from A2A AgentExecutor

- Import and inherit from `a2a.server.agent_execution.AgentExecutor`
- `execute(self, context: RequestContext, event_queue: EventQueue) -> None`:
  - Extract message from `context.message`
  - **Guard against `None`:** `context.message` returns `Message | None`; early-return with warning if `None`
  - Call `self.agent.send_message(message)`
  - Iterate over the async iterator and `event_queue.enqueue_event(event)` for each
- `cancel(self, context: RequestContext, event_queue: EventQueue) -> None`:
  - Build `TaskIdParams` from `context.task_id`; early-return with warning if `task_id` is `None`
  - Call `self.agent.cancel_task(params)`
  - Enqueue resulting task to event_queue

### 4. `sherma/entities/agent/remote.py` — Self-provisioning A2A client

- Uses proper `Client` from `a2a.client` (NOT deprecated `A2AClient`)
- Uses `ClientFactory.connect()` to create clients from URL or AgentCard
- **No more JSON-RPC wrapping/unwrapping** — `Client` handles all protocol details
- **No more `_require_client()`** — replaced with `_get_or_create_client()` that lazily creates the client
- Three ways to construct a `RemoteAgent` (in priority order):
  1. **`client`** provided → use directly
  2. **`agent_card`** provided → `ClientFactory.connect(agent_card, client_config=...)` using `get_http_client()`
  3. **`url`** provided → `ClientFactory.connect(url, client_config=...)`, then fetch card via `client.get_card()`
  4. None → raise `RuntimeError("requires either a client, agent_card, or url")`
- `url: str | None = None` — Pydantic field
- `client: Client | None = None` — proper type with `ConfigDict(arbitrary_types_allowed=True)`
- Uses `sherma.http.get_http_client()` for httpx client (respects context var)
- Client and agent_card are cached on `self` after first creation
- `send_message` — delegates to `client.send_message()` with full kwargs passthrough (`context`, `request_metadata`, `extensions`)
- `cancel_task` — delegates to `client.cancel_task()` with full kwargs passthrough (`context`, `extensions`)
- `get_card()` — delegates to `client.get_card()` after self-provisioning

### 5. `sherma/entities/agent/local.py` — Update docstring

- Reflect new method signatures: `send_message(request)` → async iterator of events, `cancel_task(request)` → Task

### 6. `sherma/langgraph/agent.py` — Update to async generator

- All imports at runtime (no `TYPE_CHECKING`)
- `send_message` becomes an async generator (yield instead of return)
- Calls `a2a_to_langgraph(request)` directly — converter accepts `Message` natively
- Calls `langgraph_to_a2a(last_message)` which returns `Message` directly — no dict intermediate
- `cancel_task` updated to accept `TaskIdParams` and return `Task`

### 7. `sherma/messages/converter.py` — Proper A2A and LangGraph types

- **`langgraph_to_a2a(message: BaseMessage) -> Message`** — accepts `BaseMessage` (from `langchain_core.messages`), returns proper `Message` object
- **`a2a_to_langgraph(message: Message) -> list[BaseMessage]`** — accepts `Message` directly, returns `list[BaseMessage]` (not `list[Any]`)
- `_content_block_to_a2a_part()` returns `Part` instead of `dict`
- `_a2a_part_to_content_block()` accepts `Part` instead of `dict`
- Imports `Message`, `Part`, `Role`, `TextPart` from `a2a.types` and `AIMessage`, `BaseMessage`, `HumanMessage` from `langchain_core.messages` at module level
- No more `_try_import_langchain()` / lazy import pattern — `langchain-core` is a required dependency
- No more dict fallback paths

### 8. `sherma/langgraph/tools.py` — Direct imports

- Import `BaseTool`, `StructuredTool` from `langchain_core.tools` at module level
- Remove `TYPE_CHECKING` guard and `try/except ImportError` fallback

### 9. `sherma/langgraph/tracing.py` — Fix langfuse import

- Correct import path: `from langfuse.langchain import CallbackHandler` (not `langfuse.callback`)
- Remove `try/except ImportError` guard — `langfuse` and `langchain` are required dependencies

### 10. `sherma/registry/agent.py` — Use `get_http_client()` and `ClientFactory.connect()`

- Uses `ClientFactory.connect(url, client_config=ClientConfig(httpx_client=...))` to create client
- Fetches card via `client.get_card()` after connecting
- Use `sherma.http.get_http_client()` instead of creating a throwaway `httpx.AsyncClient()`
- Remove `try/except ImportError` guard

### 11. `examples/weather_agent.py` — Update to use A2A types

- Construct proper `Message` object instead of dict for `send_message` input
- Iterate async events and check `isinstance(event, Message)` with attribute access (`event.parts`, `part.root.kind`, `part.root.text`) instead of dict access

### 12. Update tests

- `tests/entities/agent/test_base.py` — Use proper A2A types; `ConcreteAgent` implements full `Client` kwargs signature (`context`, `request_metadata`, `extensions`); yields `ClientEvent | Message`
- `tests/entities/agent/test_remote.py` — Mock with `AsyncMock(spec=Client)` (from `a2a.client`); mock `send_message` as async generator yielding `Message`; mock `cancel_task` returning `Task`; test no-client-or-url error; test `get_card` delegation
- `tests/a2a/test_executor.py` — `EchoAgent` implements full kwargs signature; construct `RequestContext` with `MessageSendParams(message=...)` and `task_id=...`; use real `EventQueue` and `dequeue_event(no_wait=True)` to verify enqueued events; test cancel-without-task-id raises `QueueEmpty`
- `tests/entities/agent/test_local.py` — `MyLocalAgent` implements full kwargs signature; use proper A2A types
- `tests/messages/test_converter.py` — Assert on `Message` objects (not dicts); use `HumanMessage`/`AIMessage` from langchain; verify `Role.user`/`Role.agent` and `Part` objects

## Files to Modify

1. `pyproject.toml`
2. `sherma/entities/agent/base.py`
3. `sherma/a2a/executor.py`
4. `sherma/entities/agent/remote.py`
5. `sherma/entities/agent/local.py`
6. `sherma/langgraph/agent.py`
7. `sherma/messages/converter.py`
8. `sherma/langgraph/tools.py`
9. `sherma/langgraph/tracing.py`
10. `sherma/registry/agent.py`
11. `examples/weather_agent.py`
12. `tests/entities/agent/test_base.py`
13. `tests/entities/agent/test_remote.py`
14. `tests/a2a/test_executor.py`
15. `tests/entities/agent/test_local.py`
16. `tests/messages/test_converter.py`

### 13. `sherma/a2a/executor.py` — Task creation and TaskUpdater at executor layer

- Import `TaskUpdater` from `a2a.server.tasks`, `new_task` from `a2a.utils.task`
- In `execute()`:
  1. Extract `context.current_task`; if None, create via `new_task(context.message)`
  2. Set `context.current_task = task`
  3. Create `TaskUpdater(event_queue, task.id, task.context_id)`
  4. Call `task_updater.start_work()` to signal work has begun
  5. Set `task_id` and `context_id` on the message before passing to agent
  6. Process agent responses:
     - `Message` → `task_updater.complete(message=event)`
     - `ClientEvent` (tuple) → extract artifacts via `task_updater.add_artifact()`, then update status via `task_updater.update_status()`
  7. If no events yielded, call `task_updater.complete()`
- In `cancel()`: use `TaskUpdater.cancel()` instead of directly enqueuing Task

### 14. `tests/a2a/test_executor.py` — Updated for TaskUpdater events

- Tests now expect `TaskStatusUpdateEvent` (working + completed) instead of raw `Message`/`Task`
- New tests: `test_executor_execute_creates_task`, `test_executor_execute_sets_message_ids`, `test_executor_execute_task_response`
- Cancel test expects `TaskStatusUpdateEvent` with canceled state

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright                # must be 0 errors, 0 warnings
uv run pytest -m "not integration"
uv run python examples/weather_agent.py "What is the weather in Tokyo?"
```
