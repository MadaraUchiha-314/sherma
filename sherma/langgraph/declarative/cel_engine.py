"""CEL (Common Expression Language) evaluation engine."""

from __future__ import annotations

import re
from typing import Any

import celpy
from celpy import celtypes

from sherma.exceptions import CelEvaluationError
from sherma.langgraph.declarative.cel_functions import CUSTOM_FUNCTIONS

# Concrete CEL types for isinstance checks
_CEL_CONCRETE_TYPES = (
    celtypes.BoolType,
    celtypes.IntType,
    celtypes.UintType,
    celtypes.DoubleType,
    celtypes.StringType,
    celtypes.BytesType,
    celtypes.ListType,
    celtypes.MapType,
    celtypes.TimestampType,
    celtypes.DurationType,
)

# Pattern to detect top-level ``default(expr, fallback)`` calls.
_DEFAULT_RE = re.compile(r"^default\(", re.ASCII)


def _object_to_dict(value: Any) -> dict[str, Any] | None:
    """Try to convert an arbitrary object to a dict for CEL map conversion.

    Supports Pydantic models (.model_dump()), dataclasses, and objects with __dict__.
    Returns None if no dict representation can be extracted.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[no-any-return]
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.asdict(value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
    return None


def _python_to_cel(value: Any) -> Any:
    """Convert a Python value to a CEL type."""
    if isinstance(value, _CEL_CONCRETE_TYPES):
        return value
    if value is None:
        return celtypes.BoolType(False)
    if isinstance(value, bool):
        return celtypes.BoolType(value)
    if isinstance(value, int):
        return celtypes.IntType(value)
    if isinstance(value, float):
        return celtypes.DoubleType(value)
    if isinstance(value, str):
        return celtypes.StringType(value)
    if isinstance(value, list):
        return celtypes.ListType([_python_to_cel(item) for item in value])
    if isinstance(value, dict):
        return celtypes.MapType(
            {_python_to_cel(k): _python_to_cel(v) for k, v in value.items()}
        )
    obj_dict = _object_to_dict(value)
    if obj_dict is not None:
        return _python_to_cel(obj_dict)
    return celtypes.StringType(str(value))


def _cel_to_python(value: Any) -> Any:
    """Convert a CEL type back to a Python value."""
    if isinstance(value, celtypes.BoolType):
        return bool(value)
    if isinstance(value, celtypes.IntType):
        return int(value)
    if isinstance(value, celtypes.DoubleType):
        return float(value)
    if isinstance(value, celtypes.StringType):
        return str(value)
    if isinstance(value, celtypes.ListType):
        return [_cel_to_python(item) for item in value]
    if isinstance(value, celtypes.MapType):
        return {_cel_to_python(k): _cel_to_python(v) for k, v in value.items()}
    return value


def _split_default_args(expression: str) -> tuple[str, str]:
    """Split a ``default(expr, fallback)`` call into its two argument strings.

    Handles nested parentheses, brackets, braces, and quoted strings so that
    commas inside sub-expressions are not treated as the argument separator.
    """
    # Strip the leading ``default(`` and trailing ``)``
    inner = expression[len("default(") : -1]

    depth = 0
    in_single = False
    in_double = False
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == "\\" and (in_single or in_double):
            i += 2  # skip escaped character
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                return inner[:i].strip(), inner[i + 1 :].strip()
        i += 1

    raise CelEvaluationError(
        f"CEL parse error for '{expression}': default() requires exactly two arguments"
    )


class CelEngine:
    """Wraps cel-python to evaluate CEL expressions against agent state."""

    def __init__(self, extra_vars: dict[str, Any] | None = None) -> None:
        self._env = celpy.Environment()
        self._extra_vars = extra_vars or {}

    def _build_activation(
        self,
        state: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a CEL activation from state and extra variables.

        State fields are nested under a ``state`` key so that CEL
        expressions access them via ``state.field`` or ``state["field"]``.
        Extra variables (prompts, llms, …) remain at the top level.
        *extra* provides additional one-shot variables (e.g. ``llm_response``)
        that are merged at the top level alongside ``_extra_vars``.
        """
        activation: dict[str, Any] = {}
        activation["state"] = _python_to_cel(state)
        for key, value in self._extra_vars.items():
            activation[key] = _python_to_cel(value)
        if extra:
            for key, value in extra.items():
                activation[key] = _python_to_cel(value)
        return activation

    def _evaluate_raw(
        self,
        expression: str,
        state: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Any:
        """Compile, run, and return the raw CEL result (no Python conversion)."""
        ast = self._env.compile(expression)
        prog = self._env.program(ast, functions=CUSTOM_FUNCTIONS)
        activation = self._build_activation(state, extra)
        return prog.evaluate(activation)

    def evaluate(
        self,
        expression: str,
        state: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Any:
        """Evaluate a CEL expression against state and return a Python value.

        *extra* provides additional top-level variables available in the
        expression (e.g. ``llm_response``).

        Supports a special ``default(expr, fallback)`` wrapper that returns
        *fallback* when *expr* raises an evaluation error.
        """
        try:
            if _DEFAULT_RE.match(expression):
                return self._evaluate_default(expression, state, extra)
            return _cel_to_python(self._evaluate_raw(expression, state, extra))
        except celpy.CELEvalError as exc:  # type: ignore[attr-defined]
            raise CelEvaluationError(
                f"CEL evaluation failed for '{expression}': {exc}"
            ) from exc
        except celpy.CELParseError as exc:  # type: ignore[attr-defined]
            raise CelEvaluationError(
                f"CEL parse error for '{expression}': {exc}"
            ) from exc

    def _evaluate_default(
        self,
        expression: str,
        state: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Any:
        """Handle ``default(expr, fallback)`` by trying *expr* first."""
        expr_str, fallback_str = _split_default_args(expression)
        try:
            return _cel_to_python(self._evaluate_raw(expr_str, state, extra))
        except (
            celpy.CELEvalError,  # type: ignore[attr-defined]
            celpy.CELParseError,  # type: ignore[attr-defined]
            CelEvaluationError,
        ):
            return _cel_to_python(self._evaluate_raw(fallback_str, state, extra))

    def evaluate_bool(self, expression: str, state: dict[str, Any]) -> bool:
        """Evaluate a CEL expression that must return a boolean."""
        result = self.evaluate(expression, state)
        if not isinstance(result, bool):
            raise CelEvaluationError(
                f"CEL expression '{expression}' returned {type(result).__name__}, "
                f"expected bool"
            )
        return result
