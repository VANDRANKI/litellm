"""
Helper for safe JSON loading in LiteLLM.
"""

import json
from typing import Any, Optional


def safe_json_loads(data: str, default: Optional[Any] = None) -> Any:
    """Parse a JSON string, returning *default* if parsing fails.

    This is a thin wrapper around ``json.loads`` that avoids propagating
    ``json.JSONDecodeError`` or ``ValueError`` to callers that do not care
    about the specific parse failure (e.g. when the value might be either
    JSON or a plain string).

    Args:
        data: The string to parse as JSON.
        default: Value to return when parsing fails.  Defaults to ``None``.

    Returns:
        The parsed Python object, or *default* on failure.

    Raises:
        TypeError: If *data* is not a ``str``, ``bytes``, or ``bytearray``
            (i.e. an argument type that ``json.loads`` itself would reject
            with ``TypeError``).  We intentionally let these through so
            callers catch genuine programming errors.
    """
    try:
        return json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return default
