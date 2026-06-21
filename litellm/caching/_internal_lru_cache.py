"""
LRU cache wrapper that also caches exceptions.

Python's :func:`functools.lru_cache` only caches the *return value* of
a function; if the function raises, the exception is **not** cached and
the underlying function is called again on the next invocation with the
same arguments.

:func:`lru_cache_wrapper` wraps the decorated function so that *both*
successful results and exceptions are stored in the LRU cache.  On a
cache hit the original success value is returned or the original
exception is re-raised, avoiding repeated expensive calls (e.g. HTTP
requests that are expected to fail for invalid credentials).

Internal encoding:
    The cache stores 2-tuples of ``("success", value)`` or
    ``("error", exception_instance)``.  The outer ``wrapped`` function
    unpacks the tuple and either returns the value or re-raises the
    stored exception.
"""

from functools import lru_cache
from typing import Any, Callable, Optional, Tuple, TypeVar

T = TypeVar("T")


def lru_cache_wrapper(
    maxsize: Optional[int] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory that produces an exception-caching LRU cache.

    Args:
        maxsize: Maximum number of entries to keep in the LRU cache.
            ``None`` (the default) means an unbounded cache.

    Returns:
        A decorator that, when applied to a callable, returns a new
        callable with the same signature that caches both return values
        and exceptions across repeated calls with identical arguments.

    Example::

        @lru_cache_wrapper(maxsize=128)
        def expensive_check(api_key: str) -> bool:
            # This HTTP call will only happen once per unique api_key,
            # even if it raises.
            return validate_key_with_remote_server(api_key)
    """

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @lru_cache(maxsize=maxsize)
        def wrapper(*args: Any, **kwargs: Any) -> Tuple[str, Any]:
            """Inner cached function that stores results as (status, value) tuples."""
            try:
                return ("success", f(*args, **kwargs))
            except Exception as e:
                return ("error", e)

        def wrapped(*args: Any, **kwargs: Any) -> T:
            """Unpacks the cached tuple, returning the value or re-raising the exception."""
            result: Tuple[str, Any] = wrapper(*args, **kwargs)
            if result[0] == "error":
                raise result[1]
            return result[1]  # type: ignore[return-value]

        return wrapped

    return decorator
