# Plan: Custom State Field Reducers (Replace vs. Append)

**Task:** [tasks/41-custom-state-field-reducers.md](../41-custom-state-field-reducers.md)
**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/41

---

## 1. Background & Analysis

### Current State

- `StateFieldDef` in `sherma/langgraph/declarative/schema.py` has three fields: `name`, `type`, `default`
- `_build_state_class()` in `sherma/langgraph/declarative/agent.py` checks if a field named `messages` exists:
  - **If yes:** uses `MessagesState` as base class (provides `add_messages` append reducer), skips the `messages` field annotation, adds all other fields as plain annotations (no reducer = replace)
  - **If no:** builds a plain `TypedDict` (all fields use replace)
- There is no way to:
  - Have a list field with append semantics other than `messages`
  - Have the `messages` field use replace semantics
  - Have multiple list fields with different reducer behaviors

### LangGraph Reducer Mechanics

LangGraph uses `typing.Annotated` to attach reducers to state fields:

```python
from typing import Annotated
from operator import add
from langgraph.graph import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]   # append reducer
    items: Annotated[list, add]               # simple list concat
    summary: list                             # replace (no annotation)
```

- `add_messages` — LangGraph's smart message reducer (deduplicates by ID, handles `RemoveMessage`)
- `operator.add` — simple list concatenation
- No annotation — replace semantics (default)

For this feature, `reducer: append` on a `list` field should use `add_messages` (matching current `messages` behavior), and `reducer: replace` should use no annotation.

### Key Decision: `append` uses `add_messages` for all list fields

The issue specifies "LangGraph's add-message reducer" for `append`. This is the most useful behavior since list fields in agent state are predominantly message lists. Using `operator.add` would be a weaker default. We'll use `add_messages` for `reducer: append`.

---

## 2. Implementation Steps

### Step 1: Add `reducer` field to `StateFieldDef`

**File:** `sherma/langgraph/declarative/schema.py`

- Add an optional `reducer` field: `reducer: Literal["append", "replace"] | None = None`
- `None` means "use default" — `append` for `messages`, `replace` for everything else
- Add a validator: if `reducer` is set on a non-list field, either ignore it silently or raise a validation error (prefer: ignore silently with a note in docs, matching AC: "Non-list fields ignore the `reducer` setting")

### Step 2: Update `_build_state_class()` to respect `reducer`

**File:** `sherma/langgraph/declarative/agent.py`

Current logic:
1. If `messages` field exists → use `MessagesState` base → all other fields get plain annotations
2. If no `messages` field → `TypedDict` with plain annotations

New logic:
1. Determine effective reducer for each field:
   - `field.reducer` if explicitly set
   - `"append"` if `field.name == "messages"` and `field.type == "list"` (backward compat)
   - `"replace"` otherwise
2. Build annotations dict:
   - Fields with `reducer == "append"` and `type == "list"`: `Annotated[list, add_messages]`
   - All other fields: plain type (replace semantics)
3. **No longer use `MessagesState` as base class.** Instead, always build a `TypedDict` with explicit annotations. This is cleaner and gives us full control.
   - The `messages` field with `reducer: append` gets `Annotated[list, add_messages]` — functionally equivalent to `MessagesState`
   - This removes the special-casing of `MessagesState` entirely

**Important:** Verify that switching from `MessagesState` subclass to `TypedDict` with `Annotated[list, add_messages]` doesn't break any existing behavior. `MessagesState` is defined as:
```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```
So using `Annotated[list, add_messages]` on a `TypedDict` field is functionally identical.

### Step 3: Update tests

**File:** `tests/langgraph/declarative/test_schema.py`

- Test `StateFieldDef` with `reducer` field
- Test default reducer resolution (messages → append, others → replace)
- Test that `reducer` on non-list fields is ignored

**File:** `tests/langgraph/declarative/test_agent.py`

- Add a test with a list field using `reducer: replace` alongside `messages` using `reducer: append`
- Verify that the replace-list field overwrites (not appends) on state update
- Verify that the messages field still appends

### Step 4: Update documentation

**File:** `docs/declarative-agents.md`

- Update "State Schema" section to document the `reducer` field
- Add example showing `reducer: append` and `reducer: replace` on different list fields
- Note that non-list fields always use replace regardless of `reducer` setting
- Note backward compatibility: `messages` defaults to `append`

**File:** `skills/sherma/references/declarative-agents.md`

- Mirror the same docs updates

**File:** `skills/sherma/SKILL.md`

- Update quick reference if state schema is mentioned there

---

## 3. Detailed Design

### Schema Change

```python
class StateFieldDef(BaseModel):
    """A single field in the agent state schema."""

    name: str
    type: str = "str"
    default: Any = None
    reducer: Literal["append", "replace"] | None = None
```

### State Class Builder Change

```python
def _build_state_class(agent_def, *, has_skills=False) -> type:
    from typing import Annotated, TypedDict
    from langgraph.graph import add_messages

    inject_internal = _needs_internal_state(agent_def, has_skills=has_skills)
    fields = agent_def.state.fields

    td_fields: dict[str, Any] = {}
    for field_def in fields:
        py_type = _TYPE_MAP.get(field_def.type, str)

        # Determine effective reducer
        effective_reducer = field_def.reducer
        if effective_reducer is None:
            if field_def.name == "messages" and field_def.type == "list":
                effective_reducer = "append"
            else:
                effective_reducer = "replace"

        # Apply annotation for append reducer on list fields
        if effective_reducer == "append" and field_def.type == "list":
            td_fields[field_def.name] = Annotated[py_type, add_messages]
        else:
            td_fields[field_def.name] = py_type

    if inject_internal:
        td_fields[INTERNAL_STATE_KEY] = dict

    return TypedDict("DynamicState", td_fields)
```

### Backward Compatibility

- Existing YAML without `reducer` field: `messages` list → `append` (same as before), all others → `replace` (same as before)
- No breaking changes

---

## 4. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Switching from `MessagesState` base to `TypedDict` + `Annotated` could subtly change behavior | Both are `TypedDict`-based; `MessagesState` is just a TypedDict with `Annotated[list, add_messages]`. Run all existing tests to verify. |
| `add_messages` on non-`messages` list fields may behave unexpectedly if items aren't LangChain messages | Document that `reducer: append` uses LangGraph's message reducer and is designed for message lists. |
| Users may expect `append` to do simple list concatenation | Document clearly that `append` uses `add_messages` (smart dedup by ID). |

---

## 5. Test Plan

1. **Unit: Schema validation** — `StateFieldDef` accepts `reducer: "append"`, `"replace"`, or `None`
2. **Unit: Default resolution** — `messages` list defaults to append, others default to replace
3. **Integration: Replace list** — Agent with `reducer: replace` list field; verify overwrite on update
4. **Integration: Append messages** — Existing `messages` behavior unchanged
5. **Integration: Explicit append on non-messages** — A list field with `reducer: append` accumulates
6. **Regression** — All existing tests pass unchanged
