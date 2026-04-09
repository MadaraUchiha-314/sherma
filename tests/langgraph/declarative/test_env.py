"""Tests for the env-var interpolation helper."""

from __future__ import annotations

import pytest

from sherma.exceptions import DeclarativeConfigError
from sherma.langgraph.declarative.env import expand_env_vars


def test_expand_env_vars_no_placeholder_returns_unchanged():
    assert expand_env_vars("plain string") == "plain string"
    assert expand_env_vars("") == ""


def test_expand_env_vars_single(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHERMA_TEST_VAR", "hello")
    assert expand_env_vars("${SHERMA_TEST_VAR}") == "hello"


def test_expand_env_vars_embedded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHERMA_TEST_PW", "s3cret")
    assert (
        expand_env_vars("redis://user:${SHERMA_TEST_PW}@host:6379")
        == "redis://user:s3cret@host:6379"
    )


def test_expand_env_vars_multiple(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHERMA_TEST_A", "alpha")
    monkeypatch.setenv("SHERMA_TEST_B", "beta")
    assert expand_env_vars("${SHERMA_TEST_A}-${SHERMA_TEST_B}") == "alpha-beta"


def test_expand_env_vars_default_used_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SHERMA_TEST_MISSING", raising=False)
    assert expand_env_vars("${SHERMA_TEST_MISSING:-fallback}") == "fallback"


def test_expand_env_vars_default_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SHERMA_TEST_MISSING", raising=False)
    assert expand_env_vars("${SHERMA_TEST_MISSING:-}") == ""


def test_expand_env_vars_set_var_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SHERMA_TEST_SET", "real")
    assert expand_env_vars("${SHERMA_TEST_SET:-fallback}") == "real"


def test_expand_env_vars_missing_var_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SHERMA_TEST_MISSING", raising=False)
    with pytest.raises(DeclarativeConfigError) as excinfo:
        expand_env_vars("${SHERMA_TEST_MISSING}")
    assert "SHERMA_TEST_MISSING" in str(excinfo.value)


def test_expand_env_vars_non_string_passthrough():
    assert expand_env_vars(None) is None
    assert expand_env_vars(42) == 42
    assert expand_env_vars(["a", "b"]) == ["a", "b"]
    assert expand_env_vars({"k": "v"}) == {"k": "v"}


def test_expand_env_vars_literal_dollar_without_braces_is_kept(
    monkeypatch: pytest.MonkeyPatch,
):
    # Only ``${VAR}`` form is substituted. Bare ``$VAR`` is untouched.
    monkeypatch.setenv("SHERMA_TEST_X", "yes")
    assert expand_env_vars("$SHERMA_TEST_X") == "$SHERMA_TEST_X"
