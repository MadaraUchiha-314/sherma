"""CEL (Common Expression Language) evaluation engine."""

from __future__ import annotations

from typing import Any

import celpy
from celpy import celtypes

from sherma.exceptions import CelEvaluationError

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


class CelEngine:
    """Wraps cel-python to evaluate CEL expressions against agent state."""

    def __init__(self, extra_vars: dict[str, Any] | None = None) -> None:
        self._env = celpy.Environment()
        self._extra_vars = extra_vars or {}

    def _build_activation(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build a CEL activation from state and extra variables."""
        activation: dict[str, Any] = {}
        for key, value in state.items():
            activation[key] = _python_to_cel(value)
        for key, value in self._extra_vars.items():
            activation[key] = _python_to_cel(value)
        return activation

    def evaluate(self, expression: str, state: dict[str, Any]) -> Any:
        """Evaluate a CEL expression against state and return a Python value."""
        try:
            ast = self._env.compile(expression)
            prog = self._env.program(ast)
            activation = self._build_activation(state)
            result = prog.evaluate(activation)
            return _cel_to_python(result)
        except celpy.CELEvalError as exc:  # type: ignore[attr-defined]
            raise CelEvaluationError(
                f"CEL evaluation failed for '{expression}': {exc}"
            ) from exc
        except celpy.CELParseError as exc:  # type: ignore[attr-defined]
            raise CelEvaluationError(
                f"CEL parse error for '{expression}': {exc}"
            ) from exc

    def evaluate_bool(self, expression: str, state: dict[str, Any]) -> bool:
        """Evaluate a CEL expression that must return a boolean."""
        result = self.evaluate(expression, state)
        if not isinstance(result, bool):
            raise CelEvaluationError(
                f"CEL expression '{expression}' returned {type(result).__name__}, "
                f"expected bool"
            )
        return result
