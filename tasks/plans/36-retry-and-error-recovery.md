# Plan: Retry and Error Recovery Process

**Task:** [tasks/36-retry-and-error-recovery.md](../36-retry-and-error-recovery.md)
**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/36

---

## 1. Background & Analysis

### Current State
- Every node function is wrapped in `try/except Exception` that delegates to `_run_node_error_hook()` in `nodes.py`
- The `on_node_error` hook can consume an error (set `err_ctx.error = None` → returns `{}`) or re-raise it
- There is **no** declarative way to specify retries, backoff, or fallback routing in YAML
- **Bug:** The existing `except Exception` blocks catch `GraphBubbleUp` (parent of `GraphInterrupt`), which is used by LangGraph's `interrupt()` function. This must be fixed.

### LangGraph Interrupt Exception Hierarchy
```
Exception
  └── GraphBubbleUp          ← must NEVER be caught by error handling
        └── GraphInterrupt
              └── NodeInterrupt (deprecated)
```

The `interrupt()` function from `langgraph.types` uses `GraphBubbleUp` internally. Any `except Exception` that doesn't re-raise `GraphBubbleUp` will break interrupt flow.

---

## 2. Design: Declarative `on_error` on Nodes

### YAML Schema

```yaml
nodes:
  - name: agent
    type: call_llm
    args:
      llm: { id: gpt-4o }
      prompt:
        - role: system
          content: '"You are a helpful assistant."'
        - role: messages
          content: state.messages
    on_error:
      retry:
        max_attempts: 3          # total attempts (1 initial + 2 retries)
        strategy: exponential    # "fixed" | "exponential"
        delay: 1.0               # base delay in seconds
        max_delay: 30.0          # cap for exponential backoff
      fallback: error_handler    # node to route to when retries exhausted
      # If no fallback, re-raise after retries exhausted

  - name: error_handler
    type: data_transform
    args:
      expression: |
        {
          "messages": state.messages + [state["__sherma__"]["last_error"]["message"]]
        }
```

### Schema Models (new in `schema.py`)

```python
class RetryPolicy(BaseModel):
    """Retry configuration for a node."""
    max_attempts: int = 3                           # total attempts
    strategy: Literal["fixed", "exponential"] = "exponential"
    delay: float = 1.0                              # base delay seconds
    max_delay: float = 30.0                         # max delay cap

class OnErrorDef(BaseModel):
    """Declarative error handling for a node."""
    retry: RetryPolicy | None = None
    fallback: str | None = None                     # target node name
```

Add to `NodeDef`:
```python
class NodeDef(BaseModel):
    ...
    on_error: OnErrorDef | None = None
```

### Error State Tracking

When an error occurs (after retries exhausted), store error info in `__sherma__` internal state:

```python
__sherma__["last_error"] = {
    "node": "agent",
    "type": "openai.RateLimitError",
    "message": "Rate limit exceeded",
    "attempt": 3,
}
```

This is accessible in CEL via `state["__sherma__"]["last_error"]` for downstream conditional routing and error handler nodes.

---

## 3. Implementation Steps

### Step 1: Fix Interrupt Safety (Bug Fix)

In `nodes.py`, update `_run_node_error_hook` and all `except Exception` blocks to re-raise `GraphBubbleUp` before any error handling:

```python
from langgraph.errors import GraphBubbleUp

async def _run_node_error_hook(...):
    if isinstance(exc, GraphBubbleUp):
        raise exc  # never intercept interrupt flow
    ...
```

Every node's except clause:
```python
except Exception as exc:
    if isinstance(exc, GraphBubbleUp):
        raise
    return await _run_node_error_hook(hooks, _ctx, state, exc)
```

### Step 2: Add Schema Models

- Add `RetryPolicy` and `OnErrorDef` to `schema.py`
- Add `on_error: OnErrorDef | None = None` field to `NodeDef`

### Step 3: Implement Retry Wrapper in `nodes.py`

Create a generic `_with_retry` wrapper that:

1. Reads `on_error` config from `ctx.node_def`
2. Re-raises `GraphBubbleUp` immediately (never retry interrupts)
3. On non-interrupt exceptions:
   - Retry up to `max_attempts` with configured backoff
   - Log each retry attempt
   - On exhaustion: store error in `__sherma__` internal state
   - If `fallback` is set: return a special sentinel in state that the graph routing can detect
   - If no fallback: re-raise the exception (after running `on_node_error` hook)

```python
import asyncio

async def _execute_with_retry(
    node_fn: Callable,
    ctx: NodeContext,
    state: dict[str, Any],
    hooks: HookManager | None,
) -> dict[str, Any]:
    on_error = ctx.node_def.on_error
    retry = on_error.retry if on_error else None
    max_attempts = retry.max_attempts if retry else 1

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await node_fn(ctx, state)
        except GraphBubbleUp:
            raise  # never retry interrupts
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[%s] Attempt %d/%d failed: %s",
                ctx.node_def.name, attempt, max_attempts, exc,
            )
            if attempt < max_attempts and retry:
                delay = _compute_delay(retry, attempt)
                await asyncio.sleep(delay)

    # All retries exhausted
    assert last_exc is not None
    result: dict[str, Any] = {}

    # Store error info in internal state
    if on_error:
        internal = _get_internal(state)
        internal["last_error"] = {
            "node": ctx.node_def.name,
            "type": type(last_exc).__name__,
            "message": str(last_exc),
            "attempt": max_attempts,
        }
        _set_internal(result, internal)

    # If fallback is configured, mark for routing
    if on_error and on_error.fallback:
        internal = _get_internal(state) | result.get(INTERNAL_STATE_KEY, {})
        internal["error_fallback"] = on_error.fallback
        _set_internal(result, internal)
        return result

    # No fallback — delegate to hook or re-raise
    return await _run_node_error_hook(hooks, ctx, state, last_exc)


def _compute_delay(retry: RetryPolicy, attempt: int) -> float:
    if retry.strategy == "fixed":
        return min(retry.delay, retry.max_delay)
    # exponential: delay * 2^(attempt-1)
    return min(retry.delay * (2 ** (attempt - 1)), retry.max_delay)
```

### Step 4: Wire Retry into Node Builders

Modify each node builder to use the retry wrapper instead of direct try/except. The inner function becomes the "raw" function, and the retry wrapper calls it:

```python
def build_call_llm_node(...):
    async def _raw_call_llm(_ctx, state):
        # ... existing logic without try/except ...

    async def call_llm_fn(_ctx, state):
        return await _execute_with_retry(_raw_call_llm, _ctx, state, _ctx.hook_manager)

    return partial(call_llm_fn, ctx)
```

### Step 5: Fallback Routing in Graph Compilation

In `agent.py`, when a node has `on_error.fallback`:

- After the node, inject a conditional edge that checks `__sherma__.error_fallback`
- If the sentinel is set, route to the fallback node
- Clear the sentinel after routing

This can be done in `transform.py` similar to how tool nodes are auto-injected. Or, use a simpler approach: make the fallback routing a built-in condition available in conditional edges that the transform step auto-generates.

**Simpler approach:** Wrap the node itself to handle fallback. When retries are exhausted and fallback is set, instead of returning to the normal edge target, the wrapper modifies the graph to add a conditional edge. This is done at graph build time:

For any node with `on_error.fallback`:
1. Add a conditional edge from that node with:
   - condition: `state["__sherma__"].get("error_fallback") == "<fallback_node>"`  → target: fallback node
   - default: original target (the existing edge)
2. Clear `error_fallback` in the fallback node wrapper

### Step 6: Validation

In `loader.py` `validate_config()`:
- If `on_error.fallback` references a node name, verify that node exists in the graph
- Validate `retry.max_attempts >= 1`
- Validate `retry.delay >= 0` and `retry.max_delay >= retry.delay`

### Step 7: Update Docs and Skill

- Add error handling section to `docs/README.md`
- Update `skills/sherma/references/` with new YAML schema docs
- Update `skills/sherma/SKILL.md` with `on_error` quick reference

### Step 8: Tests

- Unit tests for `_execute_with_retry` with fixed and exponential backoff
- Unit test verifying `GraphBubbleUp` is re-raised (never retried)
- Unit test for fallback routing when retries exhausted
- Unit test for error state tracking in `__sherma__`
- Integration test with a YAML agent that uses `on_error` with retry + fallback

---

## 4. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `on_error` on NodeDef, not on edges | Error handling is a property of the operation, not the routing. Keeps YAML intuitive. |
| `GraphBubbleUp` re-raise at every layer | LangGraph interrupt mechanism is exception-based; catching it breaks checkpointing. |
| Retry wrapper is generic across all node types | Avoids duplicating retry logic in each builder. Any IO-bound node benefits. |
| Error info in `__sherma__` internal state | Enables CEL-based conditional routing and error handler nodes without schema changes. |
| Fallback via conditional edge injection | Reuses existing edge infrastructure. Fallback nodes are normal nodes — no special type needed. |
| Hook integration preserved | `on_node_error` hook still fires after retries are exhausted if no fallback is configured. Hooks and declarative error handling compose. |

---

## 5. Interaction with Existing `on_node_error` Hook

The `on_node_error` hook runs **after** retries are exhausted and **only if no fallback is configured**. Order:

1. Exception occurs
2. Retry (if configured)
3. Retries exhausted → store error in `__sherma__`
4. If `fallback` → route to fallback node (hook NOT called)
5. If no fallback → call `on_node_error` hook → re-raise if not consumed

This keeps the hook as the last-resort escape hatch, while declarative `on_error` handles the common cases.

---

## Plan Revisions

_(none yet)_
