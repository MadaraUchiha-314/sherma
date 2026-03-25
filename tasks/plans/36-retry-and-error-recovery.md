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

### Node Type Support Matrix

`on_error` support is scoped per node type based on idempotency and side-effect analysis:

| Node type | `on_error.retry` | `on_error.fallback` | Rationale |
|-----------|:-:|:-:|---|
| `call_llm` | **Yes** | **Yes** | Retry wraps only `model.ainvoke()` — stateless API call, safe to retry. Primary use case. |
| `tool_node` | **No** | **Yes** | Tools can have side effects (send email, write to DB). Retry is dangerous. Fallback enables recovery routing. |
| `call_agent` | **No** | **Yes** | Sub-agent may have acted before failing. Fallback enables recovery routing. |
| `data_transform` | **No** | **No** | Pure CEL evaluation. If it fails, the expression is buggy — retry won't help. |
| `set_state` | **No** | **No** | Pure CEL evaluation. Same rationale. |
| `interrupt` | **No** | **No** | Uses `GraphBubbleUp` — error handling doesn't apply. |

Validation rejects unsupported combinations at config load time.

### Retry Scope: What Gets Retried

Retry does **not** re-execute the entire node. For `call_llm`, the retry loop wraps **only** the `model.ainvoke()` call:

```
call_llm_fn execution:
  1. node_enter hook          ← runs once
  2. Tool resolution          ← runs once
  3. Prompt construction      ← runs once
  4. before_llm_call hook     ← runs once
  5. ┌─────────────────────┐
     │ model.ainvoke()     │  ← RETRY BOUNDARY
     │ (retry loop here)   │
     └─────────────────────┘
  6. after_llm_call hook      ← runs once (on success)
  7. node_exit hook           ← runs once (on success)
```

This means:
- No idempotency requirement on hooks or prompt construction
- State is not modified between retries (same prompt, same tools)
- Only the LLM API call is retried — the expensive, failure-prone IO operation

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

  - name: fetch_data
    type: tool_node
    args: {}
    on_error:
      # retry NOT allowed on tool_node (validation error)
      fallback: handle_tool_error   # fallback only

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

### Step 3: Implement Retry in `build_call_llm_node`

Retry wraps only the `model.ainvoke()` call inside `call_llm_fn`, not the entire node:

```python
import asyncio

def _compute_delay(retry: RetryPolicy, attempt: int) -> float:
    if retry.strategy == "fixed":
        return min(retry.delay, retry.max_delay)
    # exponential: delay * 2^(attempt-1)
    return min(retry.delay * (2 ** (attempt - 1)), retry.max_delay)


# Inside build_call_llm_node → call_llm_fn:
async def call_llm_fn(_ctx, state):
    hooks = _ctx.hook_manager
    on_error = _ctx.node_def.on_error
    retry = on_error.retry if on_error else None
    max_attempts = retry.max_attempts if retry else 1

    try:
        # node_enter, tool resolution, prompt construction,
        # before_llm_call hook — all run ONCE
        ...

        # Retry loop wraps ONLY model.ainvoke()
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await model.ainvoke(all_messages)
                break  # success
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[%s] LLM attempt %d/%d failed: %s",
                    _ctx.node_def.name, attempt, max_attempts, exc,
                )
                if attempt < max_attempts and retry:
                    delay = _compute_delay(retry, attempt)
                    await asyncio.sleep(delay)
        else:
            # All retries exhausted
            raise last_exc

        # after_llm_call, node_exit — run ONCE on success
        ...
        return result

    except Exception as exc:
        if isinstance(exc, GraphBubbleUp):
            raise
        # Store error in __sherma__ if on_error configured
        if on_error:
            ...store last_error in internal state...
        # Route to fallback or delegate to hook
        if on_error and on_error.fallback:
            ...return with error_fallback sentinel...
        return await _run_node_error_hook(hooks, _ctx, state, exc)
```

### Step 4: Implement Fallback-Only for `tool_node` and `call_agent`

For `tool_node` and `call_agent`, the existing `except Exception` block is updated to check for `on_error.fallback`:

```python
except Exception as exc:
    if isinstance(exc, GraphBubbleUp):
        raise
    on_error = _ctx.node_def.on_error
    if on_error and on_error.fallback:
        # Store error info, set fallback sentinel, return
        ...
    return await _run_node_error_hook(hooks, _ctx, state, exc)
```

No retry logic — just fallback routing on error.

### Step 5: Fallback Routing in Graph Compilation

In `transform.py`, for any node with `on_error.fallback`:

1. Auto-inject a conditional edge from that node:
   - condition: checks `__sherma__.error_fallback` sentinel → routes to fallback node
   - default: original target (existing edge)
2. Clear `error_fallback` sentinel in the fallback node wrapper

### Step 6: Validation

In `loader.py` `validate_config()`:
- `on_error.retry` only allowed on `call_llm` — reject on all other node types
- `on_error.fallback` only allowed on `call_llm`, `tool_node`, `call_agent` — reject on `data_transform`, `set_state`, `interrupt`
- `on_error` entirely rejected on `interrupt` nodes
- If `on_error.fallback` references a node name, verify that node exists in the graph
- Validate `retry.max_attempts >= 1`
- Validate `retry.delay >= 0` and `retry.max_delay >= retry.delay`

### Step 7: Update Docs and Skill

- Add error handling section to `docs/README.md`
- Update `skills/sherma/references/` with new YAML schema docs
- Update `skills/sherma/SKILL.md` with `on_error` quick reference

### Step 8: Tests

- Unit test for `call_llm` retry with fixed and exponential backoff
- Unit test verifying `GraphBubbleUp` is re-raised (never retried)
- Unit test for `call_llm` fallback routing when retries exhausted
- Unit test for `tool_node` fallback (no retry)
- Unit test for `call_agent` fallback (no retry)
- Unit test for error state tracking in `__sherma__`
- Validation test: `retry` rejected on `tool_node`, `call_agent`, etc.
- Validation test: `on_error` rejected on `interrupt`, `data_transform`, `set_state`
- Integration test with a YAML agent that uses `on_error` with retry + fallback

---

## 4. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `on_error` on NodeDef, not on edges | Error handling is a property of the operation, not the routing. Keeps YAML intuitive. |
| `GraphBubbleUp` re-raise at every layer | LangGraph interrupt mechanism is exception-based; catching it breaks checkpointing. |
| Retry scoped to `call_llm` only, wraps `model.ainvoke()` | LLM API calls are stateless and safe to retry. Other node types have side effects. Retrying the full node would require idempotency guarantees. |
| Fallback allowed on IO-bound nodes (`call_llm`, `tool_node`, `call_agent`) | These are the nodes that interact with external systems and can fail at runtime. Pure nodes (`data_transform`, `set_state`) fail due to expression bugs, not transient errors. |
| `on_error` rejected on `interrupt` | Interrupt uses `GraphBubbleUp` which bypasses all error handling. Allowing `on_error` config would be misleading. |
| Error info in `__sherma__` internal state | Enables CEL-based conditional routing and error handler nodes without schema changes. |
| Fallback via conditional edge injection | Reuses existing edge infrastructure. Fallback nodes are normal nodes — no special type needed. |
| Hook integration preserved | `on_node_error` hook still fires after retries are exhausted if no fallback is configured. Hooks and declarative error handling compose. |

---

## 5. Error Handling Flow

```
                         ┌──────────────────┐
                         │   Node Executes   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Exception raised │
                         └────────┬─────────┘
                                  │
                          ┌───────▼────────┐
                          │ GraphBubbleUp?  │
                          └───┬────────┬───┘
                           yes│        │no
                              ▼        │
                     ┌────────────┐    │
                     │  Re-raise  │    │
                     │ immediately│    │
                     │ (interrupt │    │
                     │  flow)     │    │
                     └────────────┘    │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │         DECLARATIVE on_error        │
                    │           (YAML config)             │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │  on_error.retry configured?  │    │
                    │  │  (call_llm only)             │    │
                    │  └──────┬──────────────┬───────┘    │
                    │      yes│              │no          │
                    │         ▼              │            │
                    │  ┌──────────────┐      │            │
                    │  │ Retry        │      │            │
                    │  │ model.       │      │            │
                    │  │ ainvoke()    │      │            │
                    │  │ with backoff │      │            │
                    │  └──────┬───────┘      │            │
                    │         │              │            │
                    │    ┌────▼─────┐        │            │
                    │    │ Success? │        │            │
                    │    └──┬───┬───┘        │            │
                    │    yes│   │no          │            │
                    │       ▼   │(exhausted) │            │
                    │  ┌──────┐ │            │            │
                    │  │Return│ │            │            │
                    │  │result│ │            │            │
                    │  └──────┘ │            │            │
                    │           ▼            ▼            │
                    │  ┌─────────────────────────────┐    │
                    │  │ Store error in               │    │
                    │  │ __sherma__["last_error"]     │    │
                    │  └──────────────┬──────────────┘    │
                    │                 │                    │
                    │  ┌──────────────▼──────────────┐    │
                    │  │  on_error.fallback set?     │    │
                    │  │  (call_llm/tool_node/       │    │
                    │  │   call_agent only)           │    │
                    │  └──────┬──────────────┬───────┘    │
                    │      yes│              │no          │
                    │         ▼              │            │
                    │  ┌──────────────┐      │            │
                    │  │ Route to     │      │            │
                    │  │ fallback     │      │            │
                    │  │ node         │      │            │
                    │  └──────┬───────┘      │            │
                    │         │              │            │
                    │         ▼              │            │
                    │      ┌──────┐          │            │
                    │      │ DONE │          │            │
                    │      └──────┘          │            │
                    │         ▲              │            │
                    │         │              │            │
                    │  Hook NOT called       │            │
                    │  when fallback handles  │            │
                    │  the error             │            │
                    └────────────────────────┼────────────┘
                                             │
                                             ▼
                    ┌─────────────────────────────────────┐
                    │       on_node_error HOOK            │
                    │       (Python / JSON-RPC)           │
                    │                                     │
                    │  Only reached when:                 │
                    │  • No on_error configured, OR      │
                    │  • Retries exhausted AND            │
                    │    no fallback configured           │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │ Hook sets error = None?     │    │
                    │  └──────┬──────────────┬───────┘    │
                    │      yes│              │no          │
                    │         ▼              ▼            │
                    │  ┌──────────┐   ┌────────────┐     │
                    │  │ Swallow  │   │ Re-raise   │     │
                    │  │ error    │   │ exception  │     │
                    │  │ return {}│   │ (crash)    │     │
                    │  └──────────┘   └────────────┘     │
                    └─────────────────────────────────────┘
```

### Responsibility Summary

| Layer | Mechanism | Configured via | When it runs |
|-------|-----------|----------------|--------------|
| **Interrupt guard** | `isinstance(exc, GraphBubbleUp)` | Always active (not configurable) | Immediately on any exception — re-raises interrupt exceptions before any error handling |
| **Declarative `on_error`** | `on_error.retry` + `on_error.fallback` in YAML | YAML node config | First error handling layer. Retry (`call_llm` only) wraps `model.ainvoke()`. Fallback routes to recovery node. |
| **`on_node_error` hook** | Python class or JSON-RPC server | `hooks:` in YAML or programmatic | **Last resort.** Only runs if declarative `on_error` didn't handle it (no fallback configured, or no `on_error` at all). Can swallow or re-raise. |

---

## 6. Interaction with Existing `on_node_error` Hook

The `on_node_error` hook runs **after** retries are exhausted and **only if no fallback is configured**. Order:

1. Exception occurs
2. Retry `model.ainvoke()` (if `call_llm` with `retry` configured)
3. Retries exhausted → store error in `__sherma__`
4. If `fallback` → route to fallback node (hook NOT called)
5. If no fallback → call `on_node_error` hook → re-raise if not consumed

This keeps the hook as the last-resort escape hatch, while declarative `on_error` handles the common cases.

---

## Plan Revisions

### Revision 1: Scoped retry and node type restrictions

- **Changed:** Retry is no longer a generic wrapper across all node types. It is scoped to `call_llm` only, wrapping just `model.ainvoke()`.
- **Rationale:** Retrying the entire node would require idempotency guarantees. `tool_node` and `call_agent` have side effects that make blind retry dangerous. `data_transform`/`set_state` are pure CEL — failures are expression bugs, not transient errors.
- **Changed:** `on_error.fallback` is restricted to IO-bound nodes (`call_llm`, `tool_node`, `call_agent`). Rejected on `data_transform`, `set_state`, `interrupt`.
- **Changed:** `on_error` is entirely rejected on `interrupt` nodes since they use `GraphBubbleUp`.
