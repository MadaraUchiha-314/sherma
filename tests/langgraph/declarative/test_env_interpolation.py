"""Tests for ``${VAR}`` environment-variable interpolation in YAML."""

from __future__ import annotations

import pytest

from sherma.exceptions import DeclarativeConfigError
from sherma.langgraph.declarative.loader import (
    _interpolate_env_vars,
    load_declarative_config,
)


def test_substitutes_set_variable() -> None:
    result = _interpolate_env_vars(
        {"url": "${MY_URL}/path"},
        environ={"MY_URL": "https://example.com"},
    )
    assert result == {"url": "https://example.com/path"}


def test_default_used_when_var_unset() -> None:
    result = _interpolate_env_vars(
        {"url": "${MY_URL:-https://default.example.com}"},
        environ={},
    )
    assert result == {"url": "https://default.example.com"}


def test_default_ignored_when_var_set() -> None:
    result = _interpolate_env_vars(
        {"url": "${MY_URL:-fallback}"},
        environ={"MY_URL": "real"},
    )
    assert result == {"url": "real"}


def test_missing_required_var_raises_with_all_names() -> None:
    with pytest.raises(DeclarativeConfigError) as exc:
        _interpolate_env_vars(
            {"a": "${MISSING_ONE}", "b": ["${MISSING_TWO}", "${MISSING_ONE}"]},
            environ={},
        )
    msg = str(exc.value)
    assert "${MISSING_ONE}" in msg
    assert "${MISSING_TWO}" in msg


def test_recurses_through_dicts_and_lists() -> None:
    result = _interpolate_env_vars(
        {
            "outer": {
                "inner": "${A}",
                "list": ["plain", "${B}", {"deep": "${A}-${B}"}],
            }
        },
        environ={"A": "1", "B": "2"},
    )
    assert result == {
        "outer": {
            "inner": "1",
            "list": ["plain", "2", {"deep": "1-2"}],
        }
    }


def test_double_dollar_escapes_literal() -> None:
    result = _interpolate_env_vars(
        {"price": "$$5.00", "mixed": "$${KEEP} ${REPLACE}"},
        environ={"REPLACE": "yes"},
    )
    assert result == {"price": "$5.00", "mixed": "${KEEP} yes"}


def test_non_string_scalars_pass_through() -> None:
    data = {"num": 42, "flag": True, "none": None, "float": 1.5}
    assert _interpolate_env_vars(data, environ={}) == data


def test_strings_without_substitutions_pass_through() -> None:
    data = {"a": "no vars here", "b": "neither $here without braces"}
    assert _interpolate_env_vars(data, environ={}) == data


def test_lowercase_placeholders_passed_through_for_cel_template() -> None:
    """Lowercase ``${name}`` is reserved for the CEL ``template()`` function
    and must not be touched by env-var interpolation, even when the env is
    empty."""
    data = {"prompt": "hello ${available_skills} from ${user_context}"}
    assert _interpolate_env_vars(data, environ={}) == data


def test_mixed_uppercase_and_lowercase() -> None:
    """Uppercase names are interpolated; lowercase pass through untouched."""
    result = _interpolate_env_vars(
        {"text": "${HOST} -- ${user_name}"},
        environ={"HOST": "example.com"},
    )
    assert result == {"text": "example.com -- ${user_name}"}


def test_load_declarative_config_substitutes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_MODEL", "gpt-4o-mini")
    yaml_content = """\
manifest_version: 1

llms:
  - id: gpt
    version: "1.0.0"
    model_name: ${MY_MODEL}

prompts:
  - id: sys
    version: "1.0.0"
    instructions: "Be helpful"

agents:
  my-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
    graph:
      entry_point: start
      nodes:
        - name: start
          type: set_state
          args:
            values:
              x: '"hello"'
      edges: []
"""
    config = load_declarative_config(yaml_content=yaml_content)
    assert config.llms[0].model_name == "gpt-4o-mini"


def test_load_declarative_config_missing_var_raises() -> None:
    yaml_content = """\
manifest_version: 1

llms:
  - id: gpt
    version: "1.0.0"
    model_name: ${UNSET_MODEL}

prompts:
  - id: sys
    version: "1.0.0"
    instructions: "Be helpful"

agents:
  a:
    state: { fields: [{name: messages, type: list, default: []}] }
    graph:
      entry_point: s
      nodes:
        - name: s
          type: set_state
          args: { values: { x: '"hi"' } }
      edges: []
"""
    with pytest.raises(DeclarativeConfigError, match="UNSET_MODEL"):
        load_declarative_config(yaml_content=yaml_content)
