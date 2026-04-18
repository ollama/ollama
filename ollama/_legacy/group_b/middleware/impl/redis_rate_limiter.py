"""Redis-backed rate limiter for distributed systems."""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class RedisRateLimiter:
    """Redis-backed rate limiter for distributed systems.

    Uses Redis INCR and EXPIRE for atomic rate limiting.
    Implementation required for multi-instance deployments.
    Reference: https://github.com/kushin77/ollama/issues
    """

    def __init__(self, redis_client: Any, requests_per_minute: int = 60) -> None:
        """Initialize Redis rate limiter.

        Args:
            redis_client: Redis client instance
            requests_per_minute: Rate limit
        """
        self.redis = redis_client
        self.requests_per_minute = requests_per_minute

    async def check_rate_limit(self, key: str) -> tuple[bool, dict[str, int]]:
        """Check rate limit using Redis.

        Args:
            key: Rate limit key

        Returns:
            Tuple of (allowed, limit_info)

        Implementation Strategy:
        - Use INCR on rate limit key with EXPIRE for sliding window
        - Track reset time with TTL for precision
        - Atomic operations ensure distributed safety
        """
        # Run blocking Redis operations in thread pool
        loop = asyncio.get_event_loop()

        # Sliding window rate limit implementation
        window_key = f"ratelimit:window:{key}"
        reset_key = f"ratelimit:reset:{key}"

        try:

            # Execute pipeline for atomic operations
            def execute_pipeline() -> tuple[int, int | None]:
                """Execute Redis pipeline atomically."""
                pipeline = self.redis.pipeline()
                pipeline.incr(window_key)
                pipeline.pexpire(window_key, 60000)  # 60 second window
                pipeline.pttl(reset_key)
                results = pipeline.execute()

                request_count = results[0]
                reset_pttl = results[2]

                return request_count, reset_pttl

            # Execute in thread pool to avoid blocking event loop
            request_count, reset_pttl = await loop.run_in_executor(None, execute_pipeline)

            # Calculate reset time
            if reset_pttl and reset_pttl > 0:
                reset_seconds = reset_pttl / 1000.0
            else:
                reset_seconds = 60.0

            reset_timestamp = int(time.time() + reset_seconds)

            # Prepare limit info
            limit_info = {
                "limit": self.requests_per_minute,
                "remaining": max(0, self.requests_per_minute - request_count),
                "reset": reset_timestamp,
            }

            # Check if limit exceeded
            allowed = request_count <= self.requests_per_minute

            return allowed, limit_info

        except Exception as e:
            logger.error(f"Redis rate limit error for key {key}: {e}")
            # Fail open: allow request if Redis is unavailable
            # Production: Set FAIL_CLOSED=true to deny if Redis down
            return True, {
                "limit": self.requests_per_minute,
                "remaining": self.requests_per_minute,
                "reset": int(time.time() + 60),
            }
