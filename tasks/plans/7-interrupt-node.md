# Plan: Add `interrupt` Node Type to Declarative Agent

## Context

The declarative agent system supports 5 node types: `call_llm`, `tool_node`, `call_agent`, `data_transform`, `set_state`. We need to add a 6th type — `interrupt` — that calls LangGraph's `interrupt()` function to enable human-in-the-loop workflows.

LangGraph's `interrupt(value)` pauses graph execution and surfaces a value to the client. When resumed (via `Command(resume=...)`), it returns the resume value. The response should be added to `state.messages`.

## Changes

### 1. Schema — `sherma/langgraph/declarative/schema.py`

- Add `InterruptArgs` model with a `value` field (CEL expression)
- Update `NodeDef.type` Literal to include `"interrupt"`
- Update `NodeDef.args` union to include `InterruptArgs`

### 2. Node builder — `sherma/langgraph/declarative/nodes.py`

- Add `build_interrupt_node()` that:
  - Evaluates `args.value` as a CEL expression against current state
  - Calls `langgraph.types.interrupt(value)` to pause the graph
  - On resume, wraps the response as a `HumanMessage` and appends to `state.messages`

### 3. Agent graph builder — `sherma/langgraph/declarative/agent.py`

- Import and wire up `build_interrupt_node` in `_build_node()`

### 4. Validation — `sherma/langgraph/declarative/loader.py`

- Add validation: interrupt nodes require `messages` field in state

### 5. Tests

- Unit tests for `build_interrupt_node` in `test_nodes.py`
- Schema validation test in `test_schema.py`
- Integration test in `test_agent.py`

## YAML Usage Example

```yaml
nodes:
  - name: ask_name
    type: interrupt
    args:
      value: '"What is your name?"'

  - name: ask_age
    type: interrupt
    args:
      value: '"Hello " + name + ", how old are you?"'
```

## Verification

```bash
uv run pytest tests/langgraph/declarative/ -v
uv run ruff check .
uv run ruff format --check .
uv run pyright
```
