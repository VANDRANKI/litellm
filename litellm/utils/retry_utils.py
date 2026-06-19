"""Retry utilities for LiteLLM provider calls.

Provides exponential backoff helpers used by the router and direct
completion calls when transient provider errors occur.
"""

from __future__ import annotations

import time
import random
from typing import Callable, TypeVar, Any

T = TypeVar("T")


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """Calculate the delay before the next retry attempt.

    Uses exponential backoff: delay = base_delay * 2^attempt.
    Optionally adds random jitter to avoid thundering herd.

    Args:
        attempt: Zero-indexed attempt number (0 = first retry).
        base_delay: Starting delay in seconds.
        max_delay: Upper bound on delay in seconds.
        jitter: Whether to add random jitter (±25% of computed delay).

    Returns:
        Delay in seconds to wait before the next attempt.
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay *= 1 + random.uniform(-0.25, 0.25)
    return max(0.0, delay)


def is_retryable_error(exception: Exception) -> bool:
    """Return True if the exception represents a transient error worth retrying.

    Args:
        exception: The exception raised by a provider call.

    Returns:
        True for rate limits, timeouts, and service-unavailable errors.
        False for authentication, invalid-request, and not-found errors.
    """
    # Import here to avoid circular imports at module load time
    try:
        import litellm
        retryable_types = (
            litellm.exceptions.Timeout,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.RateLimitError,
        )
        return isinstance(exception, retryable_types)
    except (ImportError, AttributeError):
        return False
