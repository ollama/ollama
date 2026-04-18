"""Feature Flag Evaluator.

Evaluates feature flags based on configuration and context (user, request, time).
Supports all rollout strategies and provides performance metrics.
"""

import hashlib
from datetime import datetime

from ollama.services.feature_flags.config import (
    FeatureFlag,
    FeatureFlagConfig,
    RolloutStrategy,
)


class FeatureFlagEvaluator:
    """Evaluates whether a feature flag is enabled for a given context."""

    def __init__(self, config: FeatureFlagConfig) -> None:
        """Initialize evaluator with configuration.

        Args:
            config: Feature flag configuration containing all flags
        """
        self.config = config

    def is_enabled(
        self,
        flag_name: str,
        user_id: str | None = None,
        user_segment: str | None = None,
    ) -> bool:
        """Evaluate if feature flag is enabled for context.

        Args:
            flag_name: Name of feature flag to evaluate
            user_id: Optional user ID for user-targeting strategies
            user_segment: Optional user segment for segment-based strategies

        Returns:
            True if feature is enabled for this context, False otherwise
        """
        flag = self.config.get_flag(flag_name)
        if not flag:
            return False

        # Check if flag is globally enabled
        if not flag.enabled:
            return False

        # Check if flag has expired
        if flag.is_expired():
            return False

        # Evaluate based on strategy
        if flag.strategy == RolloutStrategy.ALL:
            return True
        elif flag.strategy == RolloutStrategy.NONE:
            return False
        elif flag.strategy == RolloutStrategy.PERCENTAGE:
            return self._evaluate_percentage(flag, user_id)
        elif flag.strategy == RolloutStrategy.GRADUAL:
            return self._evaluate_gradual(flag)
        elif flag.strategy == RolloutStrategy.USER_TARGETING:
            return self._evaluate_user_targeting(flag, user_id, user_segment)
        elif flag.strategy == RolloutStrategy.SCHEDULED:
            return self._evaluate_scheduled(flag)
        else:
            return False

    def _evaluate_percentage(self, flag: FeatureFlag, user_id: str | None) -> bool:
        """Evaluate percentage-based rollout (A/B testing)."""
        if not flag.percentage_rollout or user_id is None:
            return False

        # Create deterministic hash of user_id using SHA256
        seed = flag.percentage_rollout.seed or flag.name
        hash_input = f"{user_id}:{seed}".encode()
        hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
        percentage = (hash_value % 100) + 1

        return percentage <= flag.percentage_rollout.percentage

    def _evaluate_gradual(self, flag: FeatureFlag) -> bool:
        """Evaluate time-based gradual rollout."""
        if not flag.gradual_rollout:
            return False

        now = datetime.now(tz=None)
        rollout = flag.gradual_rollout

        # Before start date
        if now < rollout.start_date:
            return False

        # After end date
        if now >= rollout.end_date:
            return True

        # Check percentages at current time
        current_percentage = 0
        for milestone_date, milestone_percentage in rollout.percentages:
            if now >= milestone_date:
                current_percentage = milestone_percentage
            else:
                break

        return current_percentage > 0

    def _evaluate_user_targeting(
        self,
        flag: FeatureFlag,
        user_id: str | None,
        user_segment: str | None,
    ) -> bool:
        """Evaluate user segment targeting."""
        if not flag.user_targeting or not user_id:
            return False

        targeting = flag.user_targeting

        # Exclusion takes precedence
        if user_id in targeting.exclude_users:
            return False
        if user_segment and user_segment in targeting.exclude_segments:
            return False

        # Check inclusion
        user_included = not targeting.include_users or user_id in targeting.include_users
        segment_included = (
            not targeting.include_segments or user_segment in targeting.include_segments
        )

        return user_included and segment_included

    def _evaluate_scheduled(self, flag: FeatureFlag) -> bool:
        """Evaluate time window-based scheduling."""
        if not flag.scheduled_rollout:
            return False

        now = datetime.now(tz=None)
        scheduled = flag.scheduled_rollout

        # Check day of week
        if now.weekday() not in scheduled.days_of_week:
            return False

        # Check time window (simplified, assumes UTC)
        start_hour, start_min = map(int, scheduled.start_time.split(":"))
        end_hour, end_min = map(int, scheduled.end_time.split(":"))

        current_hour = now.hour
        current_min = now.minute

        start_total = start_hour * 60 + start_min
        end_total = end_hour * 60 + end_min
        current_total = current_hour * 60 + current_min

        return start_total <= current_total < end_total
