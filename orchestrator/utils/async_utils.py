"""Async helpers: timeouts, gather with error isolation, etc."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Coroutine, List, Optional, Tuple, TypeVar

T = TypeVar("T")


async def with_timeout(
    coro: Awaitable[T],
    timeout: float,
    default: Optional[T] = None,
) -> Optional[T]:
    """Run coroutine with timeout; return default on expiry instead of raising."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return default


async def gather_with_fallback(
    *coros: Awaitable[Any],
    fallbacks: Optional[List[Any]] = None,
) -> List[Any]:
    """
    Run coroutines concurrently. On individual failure, substitute the
    corresponding fallback value rather than propagating the exception.
    """
    if fallbacks is None:
        fallbacks = [None] * len(coros)

    async def safe(coro: Awaitable[Any], fallback: Any) -> Any:
        try:
            return await coro
        except Exception:
            return fallback

    return list(
        await asyncio.gather(
            *(safe(c, f) for c, f in zip(coros, fallbacks))
        )
    )


async def run_in_executor(func: Callable[..., T], *args: Any) -> T:
    """Run a synchronous blocking function in the default thread-pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


class AsyncLRUCache:
    """Simple async-safe LRU cache backed by an ordered dict."""

    def __init__(self, maxsize: int = 256) -> None:
        from collections import OrderedDict
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._maxsize = maxsize
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
