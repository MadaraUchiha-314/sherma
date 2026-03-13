# Plan: Hooks Over JSON-RPC

## Context

Sherma's hook system currently requires Python classes (imported via `import_path`). Users building declarative agents sometimes need to customize behavior via hooks but may want to implement them in a separate service. This plan adds a `RemoteHookExecutor` that communicates with an external JSON-RPC 2.0 server, allowing hooks to be implemented in any language.

## Hook Serializability Classification

| Category | Hooks | Reason |
|----------|-------|--------|
| **Fully serializable** | `before_graph_invoke`, `after_graph_invoke`, `before_skill_load`, `after_skill_load` | All fields are primitives/dicts |
| **Partially serializable** | `node_enter`, `node_exit`, `before_llm_call`, `after_llm_call`, `before_tool_call`, `after_tool_call`, `before_agent_call`, `after_agent_call`, `before_interrupt`, `after_interrupt`, `on_node_error`, `on_error` | Have `node_context` (not serializable) or `error` (BaseException) but other fields can be serialized |
| **Not feasible** | `on_chat_model_create` | Returns a Python object/callable — fundamentally cannot work over RPC |

## Serialization Strategy

- **`node_context`**: Strip on serialize, re-attach original on deserialize
- **`agent`**: Strip on serialize, re-attach original on deserialize
- **`error`** (BaseException): Serialize as `{"type": "...", "message": "..."}`, keep original on deserialize
- **`messages`**, **`response`**, **`tools`**: Serialize for observation (best-effort via LangChain `.dict()`), but keep originals on deserialize (remote cannot modify these complex objects)
- **`state`** (dict), primitive fields: Fully round-trippable — remote server can modify these
- **`chat_model`**: Hook is a no-op for remote executor

## Implementation Steps

### Step 1: Create `sherma/hooks/serialization.py`

Serialization/deserialization utilities for hook context dataclasses.

```python
# Key fields and their handling:
NON_SERIALIZABLE_FIELDS = {"node_context", "agent", "chat_model"}
ERROR_FIELDS = {"error"}
COMPLEX_FIELDS = {"messages", "response", "tools", "tool_calls"}

def serialize_context(ctx) -> dict[str, Any]:
    """Convert context dataclass to JSON-safe dict, omitting non-serializable fields."""

def deserialize_into_context(ctx_class, data, original_ctx):
    """Reconstruct context from JSON response, re-attaching non-serializable fields from original."""
```

For `serialize_context`:
- Use `dataclasses.fields()` to iterate fields
- Skip fields in `NON_SERIALIZABLE_FIELDS`
- For `error` fields: `{"type": type(e).__name__, "message": str(e)}`
- For `messages`/`response`/`tools`/`tool_calls`: try `.dict()` / list comprehension, fall back to `str()`
- Everything else: pass through as-is

For `deserialize_into_context`:
- Start with serializable fields from `data`
- Re-attach `node_context`, `agent`, `chat_model`, `error` from `original_ctx`
- For `messages`, `response`, `tools`, `tool_calls`: keep from `original_ctx` (read-only for remote)
- Construct and return `ctx_class(**merged_fields)`

### Step 2: Create `sherma/hooks/remote.py`

`RemoteHookExecutor` that extends `BaseHookExecutor`.

- Uses `httpx.AsyncClient` for HTTP transport (already a dependency)
- JSON-RPC 2.0 envelope: `{"jsonrpc": "2.0", "method": "<hook_name>", "params": {...}, "id": N}`
- `on_chat_model_create` is a no-op (returns `None`)
- On any error (network, timeout, JSON parse, RPC error response): log warning, return `None` (pass-through)
- Lazy `httpx.AsyncClient` creation, configurable timeout

```python
class RemoteHookExecutor(BaseHookExecutor):
    _UNSUPPORTED_HOOKS: ClassVar[set[str]] = {"on_chat_model_create"}

    def __init__(self, url: str, timeout: float = 30.0) -> None: ...

    async def _call_rpc(self, method: str, params: dict) -> dict | None: ...
    async def _execute_hook(self, hook_name: str, ctx: Any) -> Any: ...

    # Override all 17 hook methods to call _execute_hook()
    # on_chat_model_create remains no-op from BaseHookExecutor
```

### Step 3: Update `sherma/langgraph/declarative/schema.py`

Extend `HookDef` to support `url`:

```python
class HookDef(BaseModel):
    import_path: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _check_one_source(self) -> HookDef:
        # Exactly one of import_path or url must be set
```

### Step 4: Update `sherma/langgraph/declarative/loader.py`

Update `populate_hooks()` to handle `url`:

```python
if hook_def.url:
    from sherma.hooks.remote import RemoteHookExecutor
    hook_manager.register(RemoteHookExecutor(url=hook_def.url))
elif hook_def.import_path:
    # ... existing logic ...
```

### Step 5: Update `sherma/hooks/__init__.py`

Add `RemoteHookExecutor` to imports and `__all__`.

### Step 6: Tests

**`tests/hooks/test_serialization.py`**:
- Round-trip tests for fully serializable contexts (GraphInvokeContext, AfterGraphInvokeContext)
- Verify non-serializable fields stripped (BeforeLLMCallContext with node_context)
- Verify error fields serialized as type+message
- Verify deserialization re-attaches originals

**`tests/hooks/test_remote.py`**:
- Mock `httpx.AsyncClient.post` to test:
  - Successful RPC call returns updated context
  - `on_chat_model_create` returns `None` (no-op)
  - Network error returns `None` and logs warning
  - JSON-RPC error response returns `None`
  - Timeout returns `None`
  - `null` result returns `None`

**`tests/langgraph/declarative/test_loader.py`** (or existing test file):
- `HookDef(url="http://...")` validates
- `HookDef()` (neither field) raises
- `HookDef(import_path="...", url="...")` (both fields) raises
- `populate_hooks` with url creates `RemoteHookExecutor`

## YAML Example

```yaml
hooks:
  - import_path: my_module.MyLocalHook
  - url: http://localhost:8080/hooks
```

## Verification

1. `uv run ruff check .` — lint passes
2. `uv run ruff format --check .` — format passes
3. `uv run pyright` — type check passes
4. `uv run pytest -m "not integration"` — all unit tests pass
5. Manually verify YAML with `url` hook creates `RemoteHookExecutor` via a simple test script
