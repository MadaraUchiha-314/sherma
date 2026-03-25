# Plan: Fix args passed to interrupt (#33)

## Problem

The interrupt node calls `_find_last_ai_message(state)` first and only falls back to the CEL `args.value` when no AIMessage exists. Since `generate_response` always produces an AIMessage, the CEL value is unreachable. Users cannot send arbitrary structured metadata alongside the draft response.

## Solution

Remove `_find_last_ai_message` entirely. The interrupt node should always evaluate the CEL `args.value` expression. Users can pass the last AI message via CEL if desired (e.g., `state.messages[-1]`).

## Steps

1. **`schema.py`** — Make `InterruptArgs.value` required (`str` instead of `str | None`), update docstring.
2. **`nodes.py`** — Remove `_find_last_ai_message`. Always evaluate `cel.evaluate(args.value, state)` for the interrupt value.
3. **`loader.py`** — Remove the interrupt node from the `messages` field requirement check (interrupt no longer reads from messages automatically). Keep it for `call_llm` only.
4. **Tests (`test_nodes.py`)** — Rewrite the three interrupt tests to reflect CEL-only behavior.
5. **Integration test (`test_agent.py`)** — Update the interrupt integration test YAML if needed.
6. **Docs** — Update `docs/declarative-agents.md` and `skills/sherma/references/declarative-agents.md` to remove the AIMessage hard contract and document CEL-only behavior.
7. **Skill SKILL.md** — Update if interrupt is mentioned in quick reference or gotchas.

## Plan Revisions

_(none yet)_
