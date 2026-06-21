"""
Circular-reference-safe JSON serialisation for LiteLLM.

Python's built-in ``json.dumps`` raises ``ValueError`` on circular
references and ``TypeError`` on non-serialisable types.  This module
provides :func:`safe_dumps`, which handles both cases gracefully:

- Circular references are replaced with the sentinel string
  ``"CircularReference Detected"``.
- Objects that exceed ``max_depth`` levels of nesting are replaced with
  ``"MaxDepthExceeded"``.
- Objects that cannot be stringified at all are replaced with
  ``"Unserializable Object"``.

Pydantic ``BaseModel`` instances are serialised via ``.model_dump()``
before recursion so that field aliases and validators are respected.
"""

import json
from typing import Any, Dict, List, Set, Tuple, Union

from pydantic import BaseModel

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

# The set of types that are directly JSON-serialisable without recursion.
_PRIMITIVES = (str, int, float, bool, type(None))

# Type alias for all values that _serialize may return.
_Serialized = Union[Dict[str, Any], List[Any], Tuple[Any, ...], List[Any], str, int, float, bool, None]


def _serialize(obj: Any, seen: Set[int], depth: int, max_depth: int) -> _Serialized:
    """Recursively convert *obj* to a JSON-safe value.

    Args:
        obj: The object to serialise.
        seen: Set of ``id()`` values for objects currently on the call
            stack — used to detect circular references.
        depth: Current recursion depth.
        max_depth: Maximum allowed recursion depth before truncation.

    Returns:
        A JSON-serialisable value, or a sentinel string when the object
        cannot be represented safely.
    """
    # Check for maximum depth.
    if depth > max_depth:
        return "MaxDepthExceeded"
    # Base-case: if it is a primitive, simply return it.
    if isinstance(obj, _PRIMITIVES):
        return obj  # type: ignore[return-value]
    # Check for circular reference.
    if id(obj) in seen:
        return "CircularReference Detected"
    seen.add(id(obj))
    result: _Serialized
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(k, str):
                result[k] = _serialize(v, seen, depth + 1, max_depth)  # type: ignore[index]
        seen.remove(id(obj))
        return result
    elif isinstance(obj, list):
        result = [_serialize(item, seen, depth + 1, max_depth) for item in obj]
        seen.remove(id(obj))
        return result
    elif isinstance(obj, tuple):
        result = tuple(_serialize(item, seen, depth + 1, max_depth) for item in obj)
        seen.remove(id(obj))
        return result
    elif isinstance(obj, set):
        # sorted() returns a list, not a set, so the result type is List.
        result = sorted([_serialize(item, seen, depth + 1, max_depth) for item in obj])  # type: ignore[type-var]
        seen.remove(id(obj))
        return result
    elif isinstance(obj, BaseModel):
        dumped = obj.model_dump()
        result = _serialize(dumped, seen, depth + 1, max_depth)
        seen.remove(id(obj))
        return result
    else:
        # Fall back to string conversion for non-serializable objects.
        try:
            return str(obj)
        except Exception:
            return "Unserializable Object"


def safe_dumps(data: Any, max_depth: int = DEFAULT_MAX_RECURSE_DEPTH) -> str:
    """Recursively serialise *data* to a JSON string, handling edge cases safely.

    Unlike ``json.dumps``, this function never raises on circular references
    or non-serialisable types.  Instead it replaces problematic values with
    human-readable sentinel strings (see module docstring).

    Args:
        data: Any Python object to serialise.
        max_depth: Maximum recursion depth before truncating with
            ``"MaxDepthExceeded"``.  Defaults to
            ``litellm.constants.DEFAULT_MAX_RECURSE_DEPTH``.

    Returns:
        A JSON string representation of *data*.
    """
    safe_data = _serialize(data, set(), 0, max_depth)
    return json.dumps(safe_data, default=str)
