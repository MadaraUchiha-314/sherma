# Plan: Prompt String Templating with State Variables

## Investigation

Reviewed the [Agent Gateway CEL reference](https://agentgateway.dev/docs/standalone/latest/reference/cel/#functions-policy-all) and source code (`crates/celx/src/general.rs`, `crates/celx/src/strings.rs`). **No `template`, `format`, `sprintf`, or string interpolation function exists** in Agent Gateway's CEL. The closest are `replace`, `regexReplace`, and `+` concatenation.

## Design Decisions

- **`${key}` syntax**: Matches the issue's YAML example, familiar from shell/ES6 template literals, avoids collision with CEL string syntax.
- **Unresolved placeholders left as-is**: Safest default for multi-stage composition.
- **Values are stringified**: `str(value)` on each map value for natural int/bool coercion.
- **No `$key` (bare dollar) support**: Ambiguous with `$` in natural text. Can add later if needed.
- **No full templating engine**: Keeps everything in CEL-land without adding Jinja2/Mustache dependency.

## Steps

1. Add `cel_template(text, substitutions)` function to `sherma/langgraph/declarative/cel_functions.py`
2. Register as `"template"` in `CUSTOM_FUNCTIONS`
3. Add unit tests in `tests/langgraph/declarative/test_cel_functions.py`
4. Add integration test with prompt extra_vars in `tests/langgraph/declarative/test_cel_engine.py`
5. Update docs: `docs/declarative-agents.md`, `skills/sherma/references/declarative-agents.md`, `skills/sherma/SKILL.md`
6. Create task and plan files
