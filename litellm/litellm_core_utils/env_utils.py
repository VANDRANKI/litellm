"""
Utility helpers for reading and parsing environment variables.

All helpers in this module follow a safe-fallback pattern: if the env var
is absent or its value cannot be parsed into the requested type, the
caller-supplied *default* is returned instead of raising.  This prevents
misconfigured environments from crashing the process at import time.
"""

import os


def get_env_int(env_var: str, default: int) -> int:
    """Parse an environment variable as an integer.

    Falls back to *default* on missing, empty, or non-numeric values so
    that misconfiguration does not crash the process at import time.

    Args:
        env_var: Name of the environment variable to read.
        default: Value to return when the variable is absent or unparseable.

    Returns:
        The parsed integer value, or *default* on failure.
    """
    raw = os.getenv(env_var)
    if raw is None:
        return default
    raw = raw.strip()
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def get_env_bool(env_var: str, default: bool = False) -> bool:
    """Parse an environment variable as a boolean.

    Treats ``"1"``, ``"true"``, ``"yes"``, and ``"on"`` (all
    case-insensitive) as ``True``; all other non-empty values are
    ``False``.  Returns *default* when the variable is not set.

    This avoids the ``os.getenv(VAR, "").lower() == "true"`` pattern
    scattered across the codebase, which misses ``"1"`` / ``"yes"`` / ``"on"``.

    Args:
        env_var: Name of the environment variable to read.
        default: Value to return when the variable is absent.
            Defaults to ``False``.

    Returns:
        ``True`` if the variable is set to a recognised truthy string,
        ``False`` for any other non-empty value, or *default* when the
        variable is not set at all.

    Examples::

        # LITELLM_DEBUG=true  -> True
        # LITELLM_DEBUG=1     -> True
        # LITELLM_DEBUG=yes   -> True
        # LITELLM_DEBUG=false -> False
        # LITELLM_DEBUG=0     -> False
        # (unset)             -> False  (default)
        debug = get_env_bool("LITELLM_DEBUG")
    """
    raw = os.getenv(env_var)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
