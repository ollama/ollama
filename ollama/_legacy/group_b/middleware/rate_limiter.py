"""Rate limiting utilities and decorators.

Provides request-level and endpoint-level rate limiting with
configurable limits, windows, and strategies.
"""

import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from fastapi import Request

from ollama.exceptions import RateLimitExceededError

if TYPE_CHECKING:
    pass


class RateLimiter:
    """Token bucket rate limiter with Redis backend.

    Attributes:
        redis_client: Redis connection for distributed rate limiting
        default_limit: Default request limit per window
        default_window: Default time window in seconds
    """

    def __init__(
        self,
        redis_client: Any = None,
        redis_url: str | None = None,
        default_limit: int = 100,
        default_window: int = 60,
    ) -> None:
        """Initialize rate limiter.

        Args:
            redis_client: Redis connection (None for in-memory)
            redis_url: Optional Redis URL to connect to if client not provided
            default_limit: Default request limit per window
            default_window: Default time window in seconds
        """
        self.default_limit = default_limit
        self.default_window = default_window
        # In-memory fallback for local rate limiting
        self._local_buckets: dict[str, list[float]] = defaultdict(list)

        if redis_client:
            self.redis_client = redis_client
        elif redis_url:
            import redis

            self.redis_client = redis.from_url(redis_url)  # type: ignore[no-untyped-call]
        else:
            self.redis_client = None

    def _get_key(self, identifier: str) -> str:
        """Generate Redis key for identifier.

        Args:
            identifier: Client identifier (API key, IP, user_id)

        Returns:
            Formatted Redis key
        """
        return f"rate_limit:{identifier}"

    async def check_limit(
        self,
        identifier: str,
        limit: int | None = None,
        window: int | None = None,
    ) -> tuple[bool, int, int]:
        """Check if request is within rate limit.

        Args:
            identifier: Client identifier
            limit: Request limit (uses default if None)
            window: Time window in seconds (uses default if None)

        Returns:
            Tuple of (is_allowed, remaining, reset_time_seconds)

        Raises:
            RateLimitExceededError: If rate limit exceeded
        """
        limit = limit or self.default_limit
        window = window or self.default_window

        if self.redis_client:
            return await self._check_redis(identifier, limit, window)
        return await self._check_local(identifier, limit, window)

    async def _check_redis(self, identifier: str, limit: int, window: int) -> tuple[bool, int, int]:
        """Check rate limit using Redis backend.

        Args:
            identifier: Client identifier
            limit: Request limit
            window: Time window in seconds

        Returns:
            Tuple of (is_allowed, remaining, reset_time_seconds)
        """
        key = self._get_key(identifier)
        current_time = int(time.time())

        try:
            # Lua script for atomic rate limit check
            script = """
            local key = KEYS[1]
            local limit = tonumber(ARGV[1])
            local window = tonumber(ARGV[2])
            local current_time = tonumber(ARGV[3])
            local cutoff_time = current_time - window

            -- Remove old entries
            redis.call('ZREMRANGEBYSCORE', key, 0, cutoff_time)

            -- Count current requests
            local current = redis.call('ZCARD', key)

            if current < limit then
                -- Add new request
                redis.call('ZADD', key, current_time, current_time)
                redis.call('EXPIRE', key, window)
                return {1, limit - current - 1, window}
            else
                -- Get oldest request time for retry_after
                local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
                local retry_after = oldest[2] and (tonumber(oldest[2]) + window - current_time) or window
                return {0, 0, retry_after}
            end
            """

            rc = self.redis_client
            if rc is None:
                # defensive: fallback to local if Redis client not configured
                return await self._check_local(identifier, limit, window)

            raw = rc.eval(
                script,
                1,
                key,
                limit,
                window,
                current_time,
            )

            # Normalize result to a sequence of ints
            result_seq = cast(Sequence[Any], raw)
            allowed = bool(result_seq[0])
            remaining = int(result_seq[1])
            retry_after = int(result_seq[2])

            if not allowed:
                raise RateLimitExceededError(
                    limit=limit,
                    window=window,
                    retry_after=max(1, retry_after),
                )

            return (True, remaining, window)
        except RateLimitExceededError:
            raise
        except Exception:
            # Fallback to local rate limiting on Redis error
            return await self._check_local(identifier, limit, window)

    async def _check_local(self, identifier: str, limit: int, window: int) -> tuple[bool, int, int]:
        """Check rate limit using in-memory storage.

        Args:
            identifier: Client identifier
            limit: Request limit
            window: Time window in seconds

        Returns:
            Tuple of (is_allowed, remaining, reset_time_seconds)
        """
        current_time = time.time()
        cutoff_time = current_time - window

        # Remove old entries
        self._local_buckets[identifier] = [
            t for t in self._local_buckets[identifier] if t > cutoff_time
        ]

        current_count = len(self._local_buckets[identifier])

        if current_count < limit:
            self._local_buckets[identifier].append(current_time)
            remaining = limit - current_count - 1
            return (True, remaining, window)

        # Calculate retry_after from oldest request
        oldest = min(self._local_buckets[identifier])
        retry_after = max(1, int(oldest + window - current_time))

        raise RateLimitExceededError(
            limit=limit,
            window=window,
            retry_after=retry_after,
        )


def rate_limit(
    limit: int = 100,
    window: int = 60,
    key_func: Callable[[Request], str] | None = None,
) -> Callable[..., Any]:
    """Decorator for rate limiting endpoints.

    Args:
        limit: Request limit per window
        window: Time window in seconds
        key_func: Function to extract rate limit key from request

    Returns:
        Decorated function

    Example:
        @rate_limit(limit=10, window=60, key_func=lambda r: r.headers.get("X-API-Key"))
        async def my_endpoint(request: Request) -> dict:
            return {"status": "ok"}
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(request: Request, *args: Any, **kwargs: Any) -> Any:
            # Default key function uses client IP
            identifier = (
                key_func(request)
                if key_func
                else request.client.host if request.client else "unknown"
            )

            # Get rate limiter from app state
            rate_limiter: RateLimiter = request.app.state.rate_limiter

            # Check rate limit
            await rate_limiter.check_limit(identifier, limit, window)

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
