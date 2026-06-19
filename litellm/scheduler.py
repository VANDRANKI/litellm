"""Priority-queue based request scheduler for LiteLLM router load balancing.

The Scheduler maintains a per-model-group priority heap so that high-priority
requests are dispatched first when multiple requests are waiting for a
capacity-constrained model group.  Queue state is persisted through the
configured :class:`~litellm.caching.caching.DualCache` to support optional
Redis-backed distributed scheduling.
"""

import enum
import heapq
from typing import List, Optional

from pydantic import BaseModel

from litellm import print_verbose
from litellm.caching.caching import DualCache, RedisCache
from litellm.constants import DEFAULT_IN_MEMORY_TTL, DEFAULT_POLLING_INTERVAL


class SchedulerCacheKeys(enum.Enum):
    queue = "scheduler:queue"
    default_in_memory_ttl = (
        DEFAULT_IN_MEMORY_TTL  # cache queue in-memory for 5s when redis cache available
    )


class FlowItem(BaseModel):
    """A single request waiting in the scheduler priority queue.

    Attributes:
        priority: Dispatch priority in the range ``[0, 255]``.  **Lower values
            mean higher priority** (min-heap ordering), so ``0`` is dispatched
            first.
        request_id: Unique identifier for the request.
        model_name: The model group this request targets (used to select the
            correct per-model queue).
    """

    priority: int  # Priority between 0 and 255
    request_id: str
    model_name: str


class Scheduler:
    """Priority scheduler for LiteLLM router requests.

    Maintains a min-heap priority queue per model group.  Requests with lower
    priority values are dispatched first.  When a Redis cache is supplied the
    queue is persisted there so multiple proxy worker processes share the same
    queue state.

    Args:
        polling_interval: How frequently (in seconds) callers should poll
            :meth:`poll` to check whether their request can proceed.
            Defaults to :data:`~litellm.constants.DEFAULT_POLLING_INTERVAL`
            (3 ms).
        redis_cache: Optional Redis cache instance.  When provided the queue
            is stored in Redis instead of in-process memory, enabling
            distributed scheduling across multiple proxy workers.  In-memory
            TTL is set to a short value so the in-memory layer acts only as a
            brief read-cache.

    Attributes:
        cache: :class:`~litellm.caching.caching.DualCache` used for queue
            persistence.
        polling_interval: Configured polling interval in seconds.
    """

    cache: DualCache

    def __init__(
        self,
        polling_interval: Optional[float] = None,
        redis_cache: Optional[RedisCache] = None,
    ):
        """
        polling_interval: float or null - frequency of polling queue. Default is 3ms.
        """
        self.queue: list = []
        default_in_memory_ttl: Optional[float] = None
        if redis_cache is not None:
            # if redis-cache available frequently poll that instead of using in-memory.
            default_in_memory_ttl = SchedulerCacheKeys.default_in_memory_ttl.value
        self.cache = DualCache(
            redis_cache=redis_cache, default_in_memory_ttl=default_in_memory_ttl
        )
        self.polling_interval = (
            polling_interval or DEFAULT_POLLING_INTERVAL
        )  # default to 3ms

    async def add_request(self, request: FlowItem) -> None:
        """Insert a new request into the priority queue for its model group.

        The request is pushed onto the min-heap keyed by ``(priority,
        request_id)`` so requests with lower priority values bubble to the top.

        Args:
            request: The :class:`FlowItem` to enqueue.
        """
        # We use the priority directly, as lower values indicate higher priority
        # get the queue
        queue = await self.get_queue(model_name=request.model_name)
        # update the queue
        heapq.heappush(queue, (request.priority, request.request_id))

        # save the queue
        await self.save_queue(queue=queue, model_name=request.model_name)

    async def poll(self, id: str, model_name: str, health_deployments: list) -> bool:
        """Check whether a queued request can be dispatched immediately.

        Returns ``True`` in two situations:

        1. There are healthy deployments available for the model group (capacity
           is available, any queued request may proceed).
        2. No healthy deployments are available **but** this request is at the
           front of the queue (it is next in line and will claim capacity as
           soon as a deployment recovers).  In this case the request is popped
           from the queue.

        Returns ``False`` when no healthy deployments are available **and** this
        request is not at the front of the queue (it must wait).

        Args:
            id: The ``request_id`` of the request to check.
            model_name: The model group queue to inspect.
            health_deployments: List of currently healthy deployment configs
                returned by the router health check.

        Returns:
            ``True`` if the caller may proceed, ``False`` if it should keep
            polling.

        Raises:
            Exception: If the queue for *model_name* is empty or invalid,
                indicating a setup error.
        """
        queue = await self.get_queue(model_name=model_name)
        if not queue:
            raise Exception(
                "Incorrectly setup. Queue is invalid. Queue={}".format(queue)
            )

        # ------------
        # Setup values
        # ------------

        print_verbose(f"len(health_deployments): {len(health_deployments)}")
        if len(health_deployments) == 0:
            print_verbose(f"queue: {queue}, seeking id={id}")
            # Check if the id is at the top of the heap
            if queue[0][1] == id:
                # Remove the item from the queue
                heapq.heappop(queue)
                await self.save_queue(queue=queue, model_name=model_name)
                print_verbose(f"Popped id: {id}")
                return True
            else:
                return False

        return True

    async def remove_request(self, request_id: str, model_name: str) -> None:
        """Remove a specific request from the priority queue.

        Used when a request times out or is cancelled while waiting in the
        queue.  Rebuilds the heap invariant after the linear filter.

        Args:
            request_id: The identifier of the request to remove.
            model_name: The model group queue to update.
        """
        queue = await self.get_queue(model_name=model_name)
        filtered_queue = [item for item in queue if item[1] != request_id]
        heapq.heapify(filtered_queue)  # restore heap invariant after filtering
        await self.save_queue(queue=filtered_queue, model_name=model_name)
        print_verbose(
            f"Removed request_id: {request_id} from queue for model: {model_name}"
        )

    async def peek(self, id: str, model_name: str, health_deployments: list) -> bool:
        """Check whether *id* is at the front of the queue without consuming it.

        Unlike :meth:`poll`, this method never pops the request from the heap.
        It is intended for use when a caller wants to know its position without
        committing to dequeue.

        Args:
            id: The ``request_id`` to check.
            model_name: The model group queue to inspect.
            health_deployments: Unused; retained for API symmetry with
                :meth:`poll`.

        Returns:
            ``True`` if *id* is the highest-priority item in the queue,
            ``False`` otherwise.

        Raises:
            Exception: If the queue for *model_name* is empty or invalid.
        """
        queue = await self.get_queue(model_name=model_name)
        if not queue:
            raise Exception(
                "Incorrectly setup. Queue is invalid. Queue={}".format(queue)
            )

        # ------------
        # Setup values
        # ------------

        # Check if the id is at the top of the heap
        if queue[0][1] == id:
            return True

        return False

    def get_queue_status(self) -> list:
        """Return the raw in-process queue contents for inspection.

        Returns:
            The current in-memory queue list.  Note that when Redis is
            configured this may not reflect the authoritative queue state;
            use :meth:`get_queue` for that.
        """
        return self.queue

    async def get_queue(self, model_name: str) -> list:
        """Retrieve the priority queue for a specific model group.

        Fetches the queue from the configured cache (Redis or in-memory).
        Returns an empty list if no queue has been stored yet for this model.

        Args:
            model_name: The model group whose queue should be retrieved.

        Returns:
            A list of ``(priority, request_id)`` tuples representing the
            current heap state for *model_name*.
        """
        if self.cache is not None:
            _cache_key = "{}:{}".format(SchedulerCacheKeys.queue.value, model_name)
            response = await self.cache.async_get_cache(key=_cache_key)
            if response is None or not isinstance(response, list):
                return []
            elif isinstance(response, list):
                return response
        return self.queue

    async def save_queue(self, queue: list, model_name: str) -> None:
        """Persist the updated priority queue for a model group to the cache.

        Args:
            queue: The updated heap list (list of ``(priority, request_id)``
                tuples) to persist.
            model_name: The model group whose queue is being saved.
        """
        if self.cache is not None:
            _cache_key = "{}:{}".format(SchedulerCacheKeys.queue.value, model_name)
            await self.cache.async_set_cache(key=_cache_key, value=queue)
        return None
