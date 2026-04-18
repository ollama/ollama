"""Middleware domain.

This module provides the public API for the middleware domain,
re-exporting components from the implementation submodules.
"""

from .auth import AuthenticationMiddleware
from .impl.cache_decorator import cache_response
from .impl.cache_key import CacheKey
from .impl.cache_stats import CacheStats
from .impl.caching_middleware import CachingMiddleware
from .impl.rate_limit import RateLimiter, RateLimitMiddleware
from .impl.rate_limiter_cache import RateLimiterCache
from .impl.redis_rate_limiter import RedisRateLimiter

__all__ = [
    "AuthenticationMiddleware",
    "CacheKey",
    "CacheStats",
    "CachingMiddleware",
    "RateLimitMiddleware",
    "RateLimiter",
    "RateLimiterCache",
    "RedisRateLimiter",
    "cache_response",
]
