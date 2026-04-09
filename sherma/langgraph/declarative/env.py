"""Environment-variable interpolation for declarative config strings.

Provides :func:`expand_env_vars`, a small, reusable helper that resolves
``${VAR}`` and ``${VAR:-default}`` placeholders against ``os.environ``.

The helper is intentionally scoped to string inputs and is safe to call
on arbitrary values — non-string inputs are returned unchanged, so
pydantic ``field_validator`` hooks can use it without having to
special-case their input type.
"""

from __future__ import annotations

import os
import re
from typing import Any

from sherma.exceptions import DeclarativeConfigError

# Matches:
#   ${VAR}
#   ${VAR:-default}     (``default`` may be empty or contain any char
#                       except ``}``; no nested placeholders)
_ENV_VAR_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)


def expand_env_vars(value: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` placeholders in *value*.

    Placeholders are resolved against :data:`os.environ`.  A plain
    ``${VAR}`` that is unset raises :class:`DeclarativeConfigError`.
    ``${VAR:-default}`` falls back to ``default`` when ``VAR`` is unset
    or empty.

    Non-string inputs are returned unchanged so callers can blindly
    pipe values through this helper.
    """
    if not isinstance(value, str):
        return value

    def _replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        env_value = os.environ.get(name)
        if env_value:
            return env_value
        if default is not None:
            return default
        raise DeclarativeConfigError(
            f"Environment variable '{name}' is not set and no default "
            f"was provided (use '${{{name}:-default}}' to supply one)"
        )

    return _ENV_VAR_PATTERN.sub(_replace, value)
