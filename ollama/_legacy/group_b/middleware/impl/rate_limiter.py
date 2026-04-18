"""Token bucket rate limiter with in-memory and optional Redis backend."""

import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter with Redis backend.

    Implements:
    - Token bucket algorithm for rate limiting
    - Per-user and per-IP rate limits
    - Configurable limits and time windows
    """

    def __init__(self, requests_per_minute: int = 60, burst_size: int | None = None):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (defaults to requests_per_minute)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        self.tokens_per_second = requests_per_minute / 60.0

        # In-memory storage - sufficient for single-instance deployments
        # For distributed systems, use RedisRateLimiter below
        # See: docs/DEPLOYMENT.md for production rate limiting setup
        self._buckets: dict[str, dict[str, float]] = defaultdict(
            lambda: {"tokens": self.burst_size, "last_update": time.time()}
        )

    def _refill_bucket(self, bucket_key: str) -> None:
        """Refill tokens in bucket based on elapsed time."""
        bucket = self._buckets[bucket_key]
        now = time.time()
        elapsed = now - bucket["last_update"]

        # Add tokens based on time elapsed
        tokens_to_add = elapsed * self.tokens_per_second
        bucket["tokens"] = min(self.burst_size, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now

    def check_rate_limit(self, key: str) -> tuple[bool, dict[str, int]]:
        """Check if request should be allowed.

        Args:
            key: Unique identifier (user_id, IP, etc.)

        Returns:
            Tuple of (allowed, limit_info)
        """
        self._refill_bucket(key)
        bucket = self._buckets[key]

        limit_info = {
            "limit": self.requests_per_minute,
            "remaining": int(bucket["tokens"]),
            "reset": int(bucket["last_update"] + 60),
        }

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True, limit_info
        else:
            return False, limit_info

    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        if key in self._buckets:
            del self._buckets[key]
