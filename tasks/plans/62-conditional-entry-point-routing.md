# Plan 62: Conditional entry point routing

## Steps

1. **Update `GraphDef` in `sherma/langgraph/declarative/schema.py`**
   - Make `entry_point: str | None = None`.
   - Add a `model_validator` that requires exactly one of: `entry_point`
     set, or at least one edge in `edges` with `source == "__start__"`.

2. **Update `_build_graph` in `sherma/langgraph/declarative/agent.py`**
   - Only add the `START -> entry_point` edge when `entry_point` is set.
   - In the edges loop, translate `source == "__start__"` to LangGraph's
     `START` sentinel for both static and conditional edges.

3. **Update `validate_config` in `sherma/langgraph/declarative/loader.py`**
   - Allow `entry_point` to be `None` (skip the existence check in that case).
   - Allow edges with `source == "__start__"` to bypass the
     "edge source must be a node" check.

4. **Tests** — Add unit tests in `tests/langgraph/declarative/`:
   - Static edge from `__start__` to a node (replaces entry_point).
   - Conditional branches from `__start__` routing to different nodes.
   - Schema error when neither `entry_point` nor a `__start__` edge is set.
   - Schema error when both `entry_point` and a `__start__` edge are set.
   - Add an integration-style test building a compiled graph that branches
     at entry on a state field.

5. **Docs** — Update `docs/declarative-agents.md` (Edges section) with
   guidance that `__start__` is a valid edge source and that `entry_point`
   becomes optional in that case. Include a small example. Mirror to
   `skills/sherma/references/declarative-agents.md`.

6. **Skill SKILL.md** — Update `skills/sherma/SKILL.md` YAML quick reference
   / gotchas to mention the `__start__` edge source.

7. **Run `uv run ruff check .`, `uv run ruff format --check .`,
   `uv run pyright`, `uv run pytest -m "not integration"`.**

8. **Commit and push to
   `claude/feat-conditional-entry-point-routing-U0lJJ`.**

## Plan Revisions

_None yet._
