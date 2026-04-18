"""Feature Flag Manager.

Manages the complete lifecycle of feature flags including loading from config,
caching, metrics collection, and integration with FastAPI routes.
"""

from typing import Any

import structlog
from redis import Redis

from ollama.services.feature_flags.config import FeatureFlag, FeatureFlagConfig
from ollama.services.feature_flags.evaluator import FeatureFlagEvaluator

log = structlog.get_logger(__name__)


class FeatureFlagManager:
    """Manages feature flag configuration, evaluation, and metrics."""

    def __init__(
        self,
        config: FeatureFlagConfig,
        redis_client: Redis | None = None,
        cache_ttl_seconds: int = 300,
    ) -> None:
        """Initialize feature flag manager.

        Args:
            config: Feature flag configuration
            redis_client: Optional Redis client for caching evaluations
            cache_ttl_seconds: TTL for cached flag evaluations
        """
        self.config = config
        self.evaluator = FeatureFlagEvaluator(config)
        self.redis_client = redis_client
        self.cache_ttl_seconds = cache_ttl_seconds
        self._metrics: dict[str, dict[str, int]] = {}

    async def is_enabled(
        self,
        flag_name: str,
        user_id: str | None = None,
        user_segment: str | None = None,
    ) -> bool:
        """Check if feature flag is enabled with caching and metrics.

        Args:
            flag_name: Name of feature flag
            user_id: Optional user ID for targeting
            user_segment: Optional user segment for targeting

        Returns:
            True if feature is enabled, False otherwise
        """
        # Check cache first
        if self.redis_client:
            cache_key = self._get_cache_key(flag_name, user_id, user_segment)
            cached = await self._get_from_cache(cache_key)
            if cached is not None:
                self._record_metric(flag_name, "cache_hit")
                return cached

        # Evaluate flag
        enabled = self.evaluator.is_enabled(flag_name, user_id, user_segment)

        # Cache result
        if self.redis_client:
            await self._set_cache(cache_key, enabled)
            self._record_metric(flag_name, "cache_miss")

        # Record metric
        self._record_metric(flag_name, "evaluation")
        self._record_metric(flag_name, "enabled" if enabled else "disabled")

        log.info(
            "flag_evaluated",
            flag_name=flag_name,
            user_id=user_id,
            enabled=enabled,
        )

        return enabled

    def update_flag(self, flag: FeatureFlag) -> None:
        """Update a feature flag configuration.

        Args:
            flag: Updated feature flag configuration
        """
        self.config.add_flag(flag)
        self._invalidate_cache_for_flag(flag.name)
        log.info("flag_updated", flag_name=flag.name)

    def remove_flag(self, flag_name: str) -> None:
        """Remove a feature flag.

        Args:
            flag_name: Name of flag to remove
        """
        self.config.remove_flag(flag_name)
        self._invalidate_cache_for_flag(flag_name)
        log.info("flag_removed", flag_name=flag_name)

    def get_metrics(self, flag_name: str | None = None) -> dict[str, Any]:
        """Get metrics for feature flags.

        Args:
            flag_name: Optional specific flag name, or all if None

        Returns:
            Dictionary of metrics
        """
        if flag_name:
            return self._metrics.get(flag_name, {})
        return self._metrics.copy()

    def cleanup_expired_flags(self) -> int:
        """Remove expired feature flags.

        Returns:
            Number of flags removed
        """
        expired = self.config.list_expired_flags()
        for flag in expired:
            self.remove_flag(flag.name)
        log.info("expired_flags_cleaned", count=len(expired))
        return len(expired)

    def _get_cache_key(
        self,
        flag_name: str,
        user_id: str | None,
        user_segment: str | None,
    ) -> str:
        """Generate cache key for flag evaluation."""
        parts = ["ff", flag_name]
        if user_id:
            parts.append(f"user:{user_id}")
        if user_segment:
            parts.append(f"segment:{user_segment}")
        return ":".join(parts)

    async def _get_from_cache(self, cache_key: str) -> bool | None:
        """Get value from Redis cache."""
        if not self.redis_client:
            return None
        try:
            value = self.redis_client.get(cache_key)
            if value is not None:
                return value == b"1"
        except Exception as e:
            log.warning("cache_get_error", error=str(e), cache_key=cache_key)
        return None

    async def _set_cache(self, cache_key: str, value: bool) -> None:
        """Set value in Redis cache."""
        if not self.redis_client:
            return
        try:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl_seconds,
                b"1" if value else b"0",
            )
        except Exception as e:
            log.warning("cache_set_error", error=str(e), cache_key=cache_key)

    def _invalidate_cache_for_flag(self, flag_name: str) -> None:
        """Invalidate all cache entries for a flag."""
        if not self.redis_client:
            return
        try:
            pattern = f"ff:{flag_name}:*"
            for key in self.redis_client.scan_iter(match=pattern):
                self.redis_client.delete(key)
        except Exception as e:
            log.warning("cache_invalidate_error", error=str(e), flag_name=flag_name)

    def _record_metric(self, flag_name: str, metric_type: str) -> None:
        """Record feature flag metric."""
        if flag_name not in self._metrics:
            self._metrics[flag_name] = {}
        if metric_type not in self._metrics[flag_name]:
            self._metrics[flag_name][metric_type] = 0
        self._metrics[flag_name][metric_type] += 1
