"""Custom CEL functions extending the standard CEL library.

Provides JSON parsing, safe access, and string manipulation functions
inspired by agentgateway's CEL extensions and the cel-go strings extension.
"""

from __future__ import annotations

import json as _json
from typing import Any

from celpy import celtypes
from celpy.adapter import json_to_cel

# ---------------------------------------------------------------------------
# Tier 1: JSON functions
# ---------------------------------------------------------------------------


def cel_json(text: celtypes.StringType) -> Any:
    """Parse a JSON string into a CEL map or list.

    Usage in CEL::

        json('{"status": "ok"}')["status"]        // "ok"
        json(state.content)["items"][0]            // first item
    """
    try:
        doc = _json.loads(str(text))
    except (ValueError, TypeError) as exc:
        raise ValueError(str(exc)) from exc
    return json_to_cel(doc)


def cel_json_valid(text: celtypes.StringType) -> celtypes.BoolType:
    """Check whether a string is valid JSON.

    Usage in CEL::

        jsonValid('{"a": 1}')   // true
        jsonValid('not json')   // false
    """
    try:
        _json.loads(str(text))
    except (ValueError, TypeError):
        return celtypes.BoolType(False)
    return celtypes.BoolType(True)


# ---------------------------------------------------------------------------
# Tier 3: String extension functions (aligned with cel-go strings extension)
# ---------------------------------------------------------------------------


def cel_split(
    text: celtypes.StringType, separator: celtypes.StringType
) -> celtypes.ListType:
    """Split a string by a separator.

    Usage in CEL::

        "a,b,c".split(",")          // ["a", "b", "c"]
        state.tags.split(",")       // split state field
    """
    parts = str(text).split(str(separator))
    return celtypes.ListType([celtypes.StringType(p) for p in parts])


def cel_trim(text: celtypes.StringType) -> celtypes.StringType:
    """Strip leading and trailing whitespace.

    Usage in CEL::

        "  hello  ".trim()   // "hello"
    """
    return celtypes.StringType(str(text).strip())


def cel_lower_ascii(text: celtypes.StringType) -> celtypes.StringType:
    """Convert ASCII characters to lowercase.

    Usage in CEL::

        "HELLO".lowerAscii()   // "hello"
    """
    return celtypes.StringType(str(text).lower())


def cel_upper_ascii(text: celtypes.StringType) -> celtypes.StringType:
    """Convert ASCII characters to uppercase.

    Usage in CEL::

        "hello".upperAscii()   // "HELLO"
    """
    return celtypes.StringType(str(text).upper())


def cel_replace(
    text: celtypes.StringType,
    old: celtypes.StringType,
    new: celtypes.StringType,
) -> celtypes.StringType:
    """Replace all occurrences of a substring.

    Usage in CEL::

        "hello world".replace("world", "CEL")   // "hello CEL"
    """
    return celtypes.StringType(str(text).replace(str(old), str(new)))


def cel_index_of(
    text: celtypes.StringType, substr: celtypes.StringType
) -> celtypes.IntType:
    """Return the index of the first occurrence of a substring, or -1.

    Usage in CEL::

        "hello".indexOf("ll")   // 2
        "hello".indexOf("xyz")  // -1
    """
    return celtypes.IntType(str(text).find(str(substr)))


def cel_join(
    items: celtypes.ListType, separator: celtypes.StringType
) -> celtypes.StringType:
    """Join a list of strings with a separator.

    Usage in CEL::

        ["a", "b", "c"].join(", ")   // "a, b, c"
    """
    return celtypes.StringType(str(separator).join(str(item) for item in items))


def cel_substring(
    text: celtypes.StringType,
    start: celtypes.IntType,
    end: celtypes.IntType,
) -> celtypes.StringType:
    """Extract a substring by start (inclusive) and end (exclusive) indices.

    Usage in CEL::

        "hello world".substring(0, 5)   // "hello"
    """
    return celtypes.StringType(str(text)[int(start) : int(end)])


# ---------------------------------------------------------------------------
# Tier 5: List utilities
# ---------------------------------------------------------------------------


def cel_last(items: celtypes.ListType) -> Any:
    """Return the last element of a list.

    Raises an error if the list is empty.  Combine with ``filter()`` to
    implement a *findLast* pattern::

        last(state.messages.filter(m, m["type"] == "approval"))

    Use with ``default()`` for a safe fallback on empty results::

        default(last(state.messages.filter(m, m["type"] == "approval"))["content"], "")

    Usage in CEL::

        last([1, 2, 3])                  // 3
        last(state.items)                 // last element
        last(state.msgs.filter(m, ...))   // findLast pattern
    """
    if len(items) == 0:
        raise ValueError("last() called on empty list")
    return items[-1]


# ---------------------------------------------------------------------------
# Tier 4: Templating
# ---------------------------------------------------------------------------


def cel_template(
    text: celtypes.StringType,
    substitutions: celtypes.MapType,
) -> celtypes.StringType:
    """Substitute ``${key}`` placeholders in a string using values from a map.

    Unresolved placeholders (keys not present in the map) are left as-is.

    Usage in CEL::

        template("Hello ${name}!", {"name": "world"})   // "Hello world!"

        // With prompt instructions:
        template(prompts["plan"]["instructions"], {"skills": state.skill_list})
    """
    result = str(text)
    for key, value in substitutions.items():
        placeholder = "${" + str(key) + "}"
        result = result.replace(placeholder, str(value))
    return celtypes.StringType(result)


# ---------------------------------------------------------------------------
# Registry of all custom functions
# ---------------------------------------------------------------------------

#: Mapping of CEL function names to their Python implementations.
#: Passed to ``celpy.Environment.program(ast, functions=CUSTOM_FUNCTIONS)``.
CUSTOM_FUNCTIONS: dict[str, Any] = {
    # Tier 1: JSON
    "json": cel_json,
    "jsonValid": cel_json_valid,
    # Tier 3: String extensions
    "split": cel_split,
    "trim": cel_trim,
    "lowerAscii": cel_lower_ascii,
    "upperAscii": cel_upper_ascii,
    "replace": cel_replace,
    "indexOf": cel_index_of,
    "join": cel_join,
    "substring": cel_substring,
    # Tier 4: Templating
    "template": cel_template,
    # Tier 5: List utilities
    "last": cel_last,
}
