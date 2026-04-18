"""Rate limiting middleware stub."""

from .impl.rate_limit import RateLimitMiddleware as RateLimitImplementation

# Alias it for the test imports
RateLimitMiddleware = RateLimitImplementation
