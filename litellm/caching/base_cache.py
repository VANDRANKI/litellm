"""
Base Cache implementation. All cache implementations should inherit from this class.

Has 4 primary abstract methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache

Subclasses must implement ``async_set_cache_pipeline`` and are encouraged
to override ``batch_cache_write``, ``disconnect``, and ``test_connection``.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class BaseCache(ABC):
    """Abstract base class for LiteLLM cache backends.

    All concrete cache implementations (Redis, in-memory, S3, GCS, …)
    inherit from this class.  Sync and async variants of ``set_cache`` /
    ``get_cache`` are provided so that both sync and async call paths can
    share the same backend.

    Args:
        default_ttl: Default time-to-live in seconds for cached entries.
            Individual ``set_cache`` calls may override this via the ``ttl``
            keyword argument.
    """

    def __init__(self, default_ttl: int = 60) -> None:
        self.default_ttl = default_ttl

    def get_ttl(self, **kwargs: Any) -> Optional[int]:
        """Resolve the TTL to use for a cache write.

        Checks the ``ttl`` keyword argument first; falls back to
        ``self.default_ttl`` when it is absent or cannot be cast to ``int``.

        Args:
            **kwargs: Arbitrary cache-write kwargs.  Only ``ttl`` is
                inspected here; all other keys are ignored.

        Returns:
            TTL in seconds, or ``None`` if no TTL should be applied.
        """
        kwargs_ttl: Optional[int] = kwargs.get("ttl")
        if kwargs_ttl is not None:
            try:
                return int(kwargs_ttl)
            except ValueError:
                return self.default_ttl
        return self.default_ttl

    def set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        """Store a value in the cache under ``key``.

        Args:
            key: Cache key string.
            value: Value to cache.  Must be serializable by the backend.
            **kwargs: Backend-specific options such as ``ttl`` (int, seconds).

        Raises:
            NotImplementedError: Subclasses that do not override this method
                will raise at call time.
        """
        raise NotImplementedError

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        """Async variant of :meth:`set_cache`.

        Args:
            key: Cache key string.
            value: Value to cache.
            **kwargs: Backend-specific options (e.g. ``ttl``).

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_set_cache_pipeline(
        self, cache_list: List[Tuple[str, Any]], **kwargs: Any
    ) -> None:
        """Write multiple key-value pairs in a single round-trip where supported.

        Implementations should use a pipeline or batch write to reduce
        per-item network overhead (e.g. Redis ``MSET`` / pipeline).

        Args:
            cache_list: A list of ``(key, value)`` tuples to write.
            **kwargs: Backend-specific options (e.g. ``ttl``).
        """
        pass

    def get_cache(self, key: str, **kwargs: Any) -> Optional[Any]:
        """Retrieve a value from the cache.

        Args:
            key: Cache key to look up.
            **kwargs: Backend-specific options.

        Returns:
            The cached value, or ``None`` if the key is absent or expired.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    async def async_get_cache(self, key: str, **kwargs: Any) -> Optional[Any]:
        """Async variant of :meth:`get_cache`.

        Args:
            key: Cache key to look up.
            **kwargs: Backend-specific options.

        Returns:
            The cached value, or ``None`` if the key is absent or expired.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    async def batch_cache_write(
        self, key: str, value: Any, **kwargs: Any
    ) -> None:
        """Queue a cache write for deferred batching.

        This hook lets backends accumulate writes and flush them together
        (e.g., at the end of a request).  The default implementation
        raises ``NotImplementedError``; backends that do not support
        batching may simply delegate to :meth:`async_set_cache`.

        Args:
            key: Cache key string.
            value: Value to cache.
            **kwargs: Backend-specific options.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    async def disconnect(self) -> None:
        """Close and clean up the backend connection.

        Called during proxy shutdown.  Implementations should release
        connection pools, flush pending writes, and free any other held
        resources.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    async def test_connection(self) -> Dict[str, Any]:
        """Test the cache connection.

        Returns:
            dict: ``{"status": "success" | "failed", "message": str, "error": Optional[str]}``

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError
