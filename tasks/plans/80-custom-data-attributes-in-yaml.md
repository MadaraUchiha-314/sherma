# Plan — 80 Custom data attributes in declarative YAML

## Design

Add a top-level `data:` key to the declarative config — a free-form mapping of
static values. Surface it to CEL as the top-level `data` variable, exactly the
way `prompts`, `llms`, and `skills` are surfaced today through
`DeclarativeAgent._build_cel_extra_vars`.

```yaml
manifest_version: 1

data:
  thresholds:
    max_retries: 3
    min_confidence: 0.7
  labels:
    positive: "👍"
    negative: "👎"
  support_email: ${SUPPORT_EMAIL:-help@example.com}
```

Referenced anywhere a CEL expression is evaluated:

```yaml
# in a prompt
content: 'template(prompts["sys"]["instructions"], {"email": data.support_email})'
# in a conditional edge
condition: 'state.retry_count < data.thresholds.max_retries'
# in set_state
values: { emoji: 'data.labels[state.sentiment]' }
```

Because env-var interpolation (`_interpolate_env_vars`) walks the entire YAML
tree before Pydantic parsing, `${VAR}` inside `data` values already works.

## Steps

1. **Schema** (`sherma/langgraph/declarative/schema.py`)
   - Add `data: dict[str, Any] = Field(default_factory=dict)` to
     `DeclarativeConfig`.

2. **CEL exposure** (`sherma/langgraph/declarative/agent.py`)
   - In `_build_cel_extra_vars`, add `if config.data: extra["data"] = config.data`.

3. **Tests**
   - `tests/langgraph/declarative/test_cel_engine.py`: confirm a `data` extra var
     is reachable via dot and bracket access (engine-level).
   - `tests/langgraph/declarative/test_loader.py`: confirm `data` round-trips
     through `load_declarative_config` and that env-var interpolation applies
     inside `data` values; confirm it defaults to `{}` when omitted.
   - `tests/langgraph/declarative/test_agent.py`: build an agent whose YAML uses
     `data.*` in a `set_state`/conditional path and assert the value flows into
     state at runtime (no LLM call needed).

4. **Docs** (`docs/declarative-agents.md`)
   - Add `data:` to the top-level YAML structure listing.
   - Add a "Custom Data (`data`)" subsection under "Entity Declarations".
   - Add a "Data Binding Reference" table under "CEL Expressions" enumerating
     every exposed variable (`state`, `prompts`, `llms`, `skills`, `data`,
     `llm_response`) with scope notes — directly answering the issue's
     documentation question.
   - Update the "CEL Expressions" available-variables bullet list to include
     `data`.

5. **Skill mirror**
   - Copy the updated `docs/declarative-agents.md` to
     `skills/sherma/references/declarative-agents.md`.
   - Update `skills/sherma/SKILL.md`: add `data:` to top-level keys, add a
     `data.*` entry to the CEL cheat sheet, and note it in the gotchas/
     data-binding area.

6. **Validate**: `uv run ruff check .`, `uv run ruff format --check .`,
   `uv run pyright`, `uv run pytest -m "not integration"`.

## Plan Revisions

- (none yet)
