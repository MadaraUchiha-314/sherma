# Plan: CEL Expressive Power for Structured Data

## Context

CEL string parsing is limited — `.contains()` works for keyword routing but cannot parse structured data (JSON) from message content. Inspired by [agentgateway's CEL reference](https://agentgateway.dev/docs/standalone/latest/reference/cel/), we add custom CEL functions for JSON parsing, safe access, and string manipulation.

cel-python supports custom function registration via `Environment.program(ast, functions={...})`. The library also provides `celpy.adapter.json_to_cel()` for converting parsed Python JSON to CEL types.

## Steps

### Step 1: Create `sherma/langgraph/declarative/cel_functions.py`

Define all custom CEL functions:

**Tier 1 — JSON:**
- `json(StringType) -> MapType|ListType` — parse JSON string via `json.loads()` + `json_to_cel()`
- `jsonValid(StringType) -> BoolType` — try `json.loads()`, return bool

**Tier 2 — Safe Access:**
- `default(value, fallback) -> value` — this is special: it can't be a normal function because the first arg may error. Instead, implement as a try/except wrapper in `CelEngine.evaluate()` that catches eval errors on the first arg and falls back. Actually, since cel-python evaluates arguments eagerly before calling the function, `default()` cannot catch errors in the first argument as a regular custom function. We need a different approach:
  - Option A: Implement `default` as a special-case in `CelEngine` that rewrites the expression or catches errors.
  - Option B: Skip `default()` — users can use CEL's ternary `has(x) ? x : fallback` or `jsonValid()` guard.
  - **Decision: Option A** — wrap the entire expression evaluation in a try/except that detects `default(...)` patterns and handles them. Actually this is fragile. Better approach: since cel-python's `||` and `&&` short-circuit, users can write `jsonValid(x) && json(x)["key"] == "val"`. For safe field access, the real need is a function that doesn't error on missing keys. We'll implement `default` by evaluating the expression, and if it errors, returning the fallback. This requires special handling at the CelEngine level, not as a regular function.
  - **Revised decision: Implement `default()` as a two-pass evaluation in CelEngine** — first try evaluating the full expression; if it raises `CelEvaluationError`, check if the top-level call is `default(expr, fallback)`, and if so, evaluate just the fallback. This is clean and doesn't require AST rewriting.

**Tier 3 — String Extensions (aligned with cel-go strings extension):**
- `split(StringType, StringType) -> ListType`
- `trim(StringType) -> StringType`
- `lowerAscii(StringType) -> StringType`
- `upperAscii(StringType) -> StringType`
- `replace(StringType, StringType, StringType) -> StringType`
- `indexOf(StringType, StringType) -> IntType`
- `join(ListType, StringType) -> StringType`
- `substring(StringType, IntType, IntType) -> StringType`

Export a `CUSTOM_FUNCTIONS: dict[str, Callable]` mapping.

### Step 2: Wire into CelEngine

Modify `CelEngine.evaluate()` and `evaluate_bool()`:
- Pass `functions=CUSTOM_FUNCTIONS` to `env.program(ast, functions=...)`
- For `default()`: add a regex check — if expression matches `default(...)`, parse out the two args, try evaluating arg1, catch errors and evaluate arg2 as fallback.

### Step 3: Add tests in `tests/langgraph/declarative/test_cel_functions.py`

Test each function:
- `json()`: valid object, array, nested, primitives, invalid JSON raises error
- `jsonValid()`: true for valid, false for invalid
- `default()`: returns value when no error, returns fallback on error
- `split()`, `trim()`, `lowerAscii()`, `upperAscii()`, `replace()`, `indexOf()`, `join()`, `substring()`
- Method-style invocation (e.g., `state.text.split(",")`)
- Integration with message content: `json(state.messages[0]["content"])["status"]`

### Step 4: Update docs and skill

- `docs/declarative-agents.md`: Add "Custom Functions" section under CEL Expressions with tables and examples for all tiers
- `skills/sherma/SKILL.md`: Update CEL Cheat Sheet with custom function patterns
- `skills/sherma/references/declarative-agents.md`: Copy of docs version

### Step 5: Run checks

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run pyright`
- `uv run pytest`
