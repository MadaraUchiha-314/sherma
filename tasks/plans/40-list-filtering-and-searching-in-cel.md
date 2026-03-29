# Plan: List Filtering and Searching in CEL

GitHub Issue: https://github.com/MadaraUchiha-314/sherma/issues/40

## Research Findings

### cel-python built-in macros (already working)

cel-python v0.5.0 ships with **standard CEL macros** that already work through sherma's `CelEngine`:

| Macro | Syntax | Description |
|-------|--------|-------------|
| `filter` | `list.filter(x, predicate)` | Returns elements matching the predicate |
| `exists` | `list.exists(x, predicate)` | Returns `true` if any element matches |
| `all` | `list.all(x, predicate)` | Returns `true` if all elements match |
| `exists_one` | `list.exists_one(x, predicate)` | Returns `true` if exactly one element matches |
| `map` | `list.map(x, expr)` | Transforms each element |

These are **macros** (not functions), meaning they support variable binding and predicate expressions natively. No registration in `CUSTOM_FUNCTIONS` is needed — they are part of the CEL language spec.

**Verified working examples:**
```cel
state.messages.filter(m, m["type"] == "human")
state.messages.exists(m, m["type"] == "ai")
state.items.all(x, x > 0)
state.items.map(x, x * 2)
size(state.messages.filter(m, m["type"] == "human")) > 0
```

### What's missing: `findLast`

`findLast(list, var, predicate)` is **not** a standard CEL macro and cannot be implemented as a simple custom function because cel-python custom functions don't support macro-style variable binding (the `var, predicate` pattern).

**Solution:** Provide a `last(list)` custom function that returns the last element of a list (or `null`/error on empty). Users combine it with the built-in `filter` macro:

```cel
# findLast equivalent:
last(state.messages.filter(m, m["type"] == "approval_decision"))

# With default() for safe fallback:
default(last(state.messages.filter(m, m["type"] == "approval"))["content"], "")
```

This is idiomatic CEL (composing primitives) and avoids fighting the cel-python architecture.

## Implementation Steps

### Step 1: Add `last()` custom function

**File:** `sherma/langgraph/declarative/cel_functions.py`

Add a new section (Tier 5: List utilities) with:
- `cel_last(items: celtypes.ListType) -> Any` — returns the last element of a list, raises error if empty
- Register as `"last"` in `CUSTOM_FUNCTIONS`

### Step 2: Add tests for built-in macros

**File:** `tests/langgraph/declarative/test_cel_functions.py`

Add a new test class `TestListMacros` covering:
- `filter` with primitives and maps (message-like dicts)
- `exists` returning true/false
- `all` returning true/false
- `exists_one`
- `map` transformation
- `size` + `filter` combo (from the issue example)
- Nested predicate access (e.g., `m["additional_kwargs"]["type"]`)

### Step 3: Add tests for `last()` function

**File:** `tests/langgraph/declarative/test_cel_functions.py`

Add `TestLast` class:
- `last` on non-empty list
- `last` on single-element list
- `last` on empty list (raises error)
- `last` combined with `filter` (the `findLast` pattern)
- `last` with `default()` fallback on empty filter result

### Step 4: Add integration tests

**File:** `tests/langgraph/declarative/test_cel_functions.py`

In `TestIntegration`, add tests combining the new features:
- `filter` + `last` + `default` (the full `findLast` pattern from the issue)
- `exists` on LangChain message objects
- `filter` on messages with `additional_kwargs` access

### Step 5: Update docs

**File:** `docs/declarative-agents.md`

In the "CEL Expressions" → "Custom Functions" section, add:
- A new "List Macros (built-in)" subsection documenting `filter`, `exists`, `all`, `exists_one`, `map`
- A new "List Utilities" subsection documenting `last()`
- A "Common Patterns" subsection showing the `findLast` equivalent pattern
- Update the examples section with list filtering examples from the issue

### Step 6: Update skill references

**Files:** `skills/sherma/` — copy updated docs references and update `SKILL.md` if it contains CEL cheat sheet or function listings.

### Step 7: Run tests and lint

```bash
uv run pytest tests/langgraph/declarative/test_cel_functions.py -v
uv run pytest tests/langgraph/declarative/test_cel_engine.py -v
uv run ruff check .
uv run ruff format --check .
uv run pyright
```
