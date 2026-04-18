"""Feature Flag Configuration Models.

Defines the structure and validation for feature flag definitions, including:
- Flag metadata (name, description, owner)
- Rollout strategies (percentage, gradual, user targeting)
- Evaluation rules (conditions, fallback behavior)
- Monitoring (metrics tracking, alerting)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RolloutStrategy(str, Enum):
    """Strategies for feature flag rollout.

    Attributes:
        ALL: Feature enabled for all users
        NONE: Feature disabled for all users
        PERCENTAGE: Enabled for percentage of users (A/B testing)
        GRADUAL: Progressive rollout over time
        USER_TARGETING: Enabled for specific users/segments
        SCHEDULED: Enabled during specific time window
    """

    ALL = "all"
    NONE = "none"
    PERCENTAGE = "percentage"
    GRADUAL = "gradual"
    USER_TARGETING = "user_targeting"
    SCHEDULED = "scheduled"


@dataclass
class PercentageRollout:
    """Percentage-based rollout configuration (A/B testing).

    Attributes:
        percentage: Percentage of users (0-100) to enable feature for
        seed: Optional seed for deterministic rollout (user_id hashed with seed)
    """

    percentage: int = field(default=0)
    seed: str | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate percentage range."""
        if not 0 <= self.percentage <= 100:
            raise ValueError(f"Percentage must be 0-100, got {self.percentage}")


@dataclass
class GradualRollout:
    """Time-based gradual rollout configuration.

    Attributes:
        start_date: When rollout begins (UTC)
        end_date: When rollout completes (UTC)
        percentages: List of milestones [(date, percentage), ...]
    """

    start_date: datetime = field()
    end_date: datetime = field()
    percentages: list[tuple[datetime, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate rollout timeline."""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")


@dataclass
class UserTargeting:
    """User segment targeting configuration.

    Attributes:
        include_users: List of user IDs to include
        exclude_users: List of user IDs to exclude
        include_segments: List of user segments to include (e.g., "beta", "enterprise")
        exclude_segments: List of user segments to exclude
    """

    include_users: list[str] = field(default_factory=list)
    exclude_users: list[str] = field(default_factory=list)
    include_segments: list[str] = field(default_factory=list)
    exclude_segments: list[str] = field(default_factory=list)


@dataclass
class ScheduledRollout:
    """Time window-based rollout configuration.

    Attributes:
        start_time: Start time in HH:MM format (24-hour)
        end_time: End time in HH:MM format (24-hour)
        timezone: IANA timezone (e.g., "US/Eastern")
        days_of_week: List of days to enable (0=Monday, 6=Sunday)
    """

    start_time: str = field(default="09:00")
    end_time: str = field(default="17:00")
    timezone: str = field(default="UTC")
    days_of_week: list[int] = field(default_factory=lambda: list(range(5)))  # Mon-Fri


@dataclass
class FeatureFlag:
    """Feature flag configuration.

    Attributes:
        name: Unique flag identifier (snake_case)
        description: Human-readable description
        enabled: Global enable/disable toggle
        strategy: Rollout strategy (ALL, NONE, PERCENTAGE, etc.)
        percentage_rollout: Percentage-based rollout config
        gradual_rollout: Time-based gradual rollout config
        user_targeting: User segment targeting config
        scheduled_rollout: Time window-based config
        owner: Team/person responsible for flag
        created_at: Flag creation timestamp
        expires_at: Optional expiration date (for temporary flags)
        metrics: List of metrics to track (e.g., ["conversion_rate", "latency"])
        alert_threshold: Performance threshold for alerting
        rollback_on_alert: Auto-rollback if metrics exceed threshold
    """

    name: str = field()
    description: str = field(default="")
    enabled: bool = field(default=False)
    strategy: RolloutStrategy = field(default=RolloutStrategy.NONE)
    percentage_rollout: PercentageRollout | None = field(default=None)
    gradual_rollout: GradualRollout | None = field(default=None)
    user_targeting: UserTargeting | None = field(default=None)
    scheduled_rollout: ScheduledRollout | None = field(default=None)
    owner: str = field(default="")
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = field(default=None)
    metrics: list[str] = field(default_factory=list)
    alert_threshold: dict[str, float] | None = field(default=None)
    rollback_on_alert: bool = field(default=True)

    def __post_init__(self) -> None:
        """Validate feature flag configuration."""
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"Invalid flag name: {self.name}")

        if self.expires_at and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")

        # Validate strategy-specific config exists
        if self.strategy == RolloutStrategy.PERCENTAGE and not self.percentage_rollout:
            raise ValueError("percentage_rollout required for PERCENTAGE strategy")
        if self.strategy == RolloutStrategy.GRADUAL and not self.gradual_rollout:
            raise ValueError("gradual_rollout required for GRADUAL strategy")
        if self.strategy == RolloutStrategy.USER_TARGETING and not self.user_targeting:
            raise ValueError("user_targeting required for USER_TARGETING strategy")
        if self.strategy == RolloutStrategy.SCHEDULED and not self.scheduled_rollout:
            raise ValueError("scheduled_rollout required for SCHEDULED strategy")

    def is_expired(self) -> bool:
        """Check if flag has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at


@dataclass
class FeatureFlagConfig:
    """Complete feature flag configuration set.

    Attributes:
        flags: Dictionary of flag_name -> FeatureFlag
        default_strategy: Default rollout strategy for new flags
        metrics_enabled: Enable metrics collection
        cache_ttl_seconds: Redis cache TTL for evaluated flags
    """

    flags: dict[str, FeatureFlag] = field(default_factory=dict)
    default_strategy: RolloutStrategy = field(default=RolloutStrategy.NONE)
    metrics_enabled: bool = field(default=True)
    cache_ttl_seconds: int = field(default=300)

    def add_flag(self, flag: FeatureFlag) -> None:
        """Add or update a feature flag."""
        self.flags[flag.name] = flag

    def remove_flag(self, flag_name: str) -> None:
        """Remove a feature flag."""
        if flag_name in self.flags:
            del self.flags[flag_name]

    def get_flag(self, flag_name: str) -> FeatureFlag | None:
        """Get a feature flag by name."""
        return self.flags.get(flag_name)

    def list_active_flags(self) -> list[FeatureFlag]:
        """List all active (non-expired) flags."""
        return [f for f in self.flags.values() if not f.is_expired()]

    def list_expired_flags(self) -> list[FeatureFlag]:
        """List all expired flags for cleanup."""
        return [f for f in self.flags.values() if f.is_expired()]
