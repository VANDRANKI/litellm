"""Typed retry policy utilities for LiteLLM.

Provides configurable exponential-backoff retry logic used by the router
and direct completion callers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence, Type


@dataclass
class RetryPolicy:
    """Configuration for retry behaviour on transient failures.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Upper bound on delay between retries.
        backoff_factor: Multiplier applied to delay after each failure.
        retryable_exceptions: Exception types that should trigger a retry.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retryable_exceptions: tuple[Type[BaseException], ...] = field(
        default_factory=lambda: (Exception,)
    )

    def delay_for(self, attempt: int) -> float:
        """Return the sleep duration for a given attempt number (0-indexed)."""
        delay = self.base_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


def with_retry(
    fn: Callable[[], object],
    policy: RetryPolicy | None = None,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> object:
    """Execute *fn* with retry logic governed by *policy*.

    Args:
        fn: Zero-argument callable to attempt.
        policy: Retry configuration; uses sensible defaults when omitted.
        on_retry: Optional hook called with (attempt_number, exception) before
            each sleep so callers can log or instrument retries.

    Returns:
        The return value of *fn* on success.

    Raises:
        The last exception raised by *fn* after all retries are exhausted.
    """
    if policy is None:
        policy = RetryPolicy()

    last_exc: BaseException | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            return fn()
        except policy.retryable_exceptions as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt == policy.max_retries:
                break
            if on_retry is not None:
                on_retry(attempt, exc)
            time.sleep(policy.delay_for(attempt))

    raise last_exc  # type: ignore[misc]


DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
)

AGGRESSIVE_RETRY_POLICY = RetryPolicy(
    max_retries=5,
    base_delay=0.5,
    max_delay=60.0,
    backoff_factor=2.0,
)
