"""Unit tests for Feature Flags System.

Comprehensive test suite covering configuration, evaluation, and management.
"""

from datetime import datetime, timedelta

import pytest

from ollama.services.feature_flags.config import (
    FeatureFlag,
    FeatureFlagConfig,
    GradualRollout,
    PercentageRollout,
    RolloutStrategy,
    ScheduledRollout,
    UserTargeting,
)
from ollama.services.feature_flags.evaluator import FeatureFlagEvaluator
from ollama.services.feature_flags.manager import FeatureFlagManager


class TestPercentageRollout:
    """Tests for percentage-based A/B testing rollout."""

    def test_percentage_validation(self) -> None:
        """Percentage must be 0-100."""
        with pytest.raises(ValueError):
            PercentageRollout(percentage=101)

        with pytest.raises(ValueError):
            PercentageRollout(percentage=-1)

        PercentageRollout(percentage=50)  # Should not raise

    def test_percentage_with_seed(self) -> None:
        """Seed allows custom deterministic bucketing."""
        rollout = PercentageRollout(percentage=50, seed="my-seed")
        assert rollout.seed == "my-seed"


class TestGradualRollout:
    """Tests for time-based gradual rollout."""

    def test_gradual_date_validation(self) -> None:
        """Start date must be before end date."""
        now = datetime.now(tz=None)
        with pytest.raises(ValueError):
            GradualRollout(
                start_date=now + timedelta(days=1),
                end_date=now,
            )

    def test_gradual_with_milestones(self) -> None:
        """Define progressive rollout milestones."""
        now = datetime.now(tz=None)
        rollout = GradualRollout(
            start_date=now,
            end_date=now + timedelta(days=14),
            percentages=[
                (now + timedelta(days=0), 10),
                (now + timedelta(days=7), 50),
                (now + timedelta(days=14), 100),
            ],
        )
        assert len(rollout.percentages) == 3
        assert rollout.percentages[0] == (now + timedelta(days=0), 10)


class TestUserTargeting:
    """Tests for user segment targeting."""

    def test_user_inclusion(self) -> None:
        """Include specific users."""
        targeting = UserTargeting(include_users=["user_1", "user_2"])
        assert "user_1" in targeting.include_users

    def test_segment_inclusion(self) -> None:
        """Include user segments like beta, enterprise."""
        targeting = UserTargeting(include_segments=["beta", "enterprise"])
        assert "beta" in targeting.include_segments

    def test_user_exclusion(self) -> None:
        """Exclude specific users overrides inclusion."""
        targeting = UserTargeting(
            include_users=["user_1", "user_2"],
            exclude_users=["user_2"],
        )
        assert "user_2" in targeting.exclude_users


class TestScheduledRollout:
    """Tests for time window-based rollout."""

    def test_default_business_hours(self) -> None:
        """Default scheduling is business hours Mon-Fri."""
        scheduled = ScheduledRollout()
        assert scheduled.start_time == "09:00"
        assert scheduled.end_time == "17:00"
        assert scheduled.days_of_week == [0, 1, 2, 3, 4]  # Mon-Fri

    def test_custom_schedule(self) -> None:
        """Custom time windows and days."""
        scheduled = ScheduledRollout(
            start_time="08:00",
            end_time="18:00",
            days_of_week=[0, 1, 2, 3, 4, 5],  # Mon-Sat
        )
        assert scheduled.start_time == "08:00"
        assert scheduled.days_of_week == [0, 1, 2, 3, 4, 5]


class TestFeatureFlag:
    """Tests for feature flag configuration."""

    def test_flag_creation(self) -> None:
        """Create feature flag with valid configuration."""
        flag = FeatureFlag(
            name="test_flag",
            description="Test flag",
            enabled=True,
            strategy=RolloutStrategy.ALL,
        )
        assert flag.name == "test_flag"
        assert not flag.is_expired()

    def test_flag_expiration(self) -> None:
        """Flag expires after expiration date."""
        flag = FeatureFlag(
            name="temp_flag",
            enabled=True,
            expires_at=datetime.now(tz=None) - timedelta(days=1),
        )
        assert flag.is_expired()

    def test_flag_name_validation(self) -> None:
        """Flag name must be valid."""
        with pytest.raises(ValueError):
            FeatureFlag(name="")  # Empty name

        with pytest.raises(ValueError):
            FeatureFlag(name="invalid-name!")  # Invalid characters

    def test_flag_strategy_validation(self) -> None:
        """Strategy must have matching config."""
        with pytest.raises(ValueError):
            FeatureFlag(
                name="test",
                strategy=RolloutStrategy.PERCENTAGE,
                # Missing percentage_rollout
            )

        # Should not raise with matching config
        FeatureFlag(
            name="test",
            strategy=RolloutStrategy.PERCENTAGE,
            percentage_rollout=PercentageRollout(percentage=50),
        )


class TestFeatureFlagConfig:
    """Tests for feature flag configuration management."""

    def test_add_flag(self) -> None:
        """Add flag to configuration."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(name="test", strategy=RolloutStrategy.ALL)
        config.add_flag(flag)
        assert config.get_flag("test") == flag

    def test_remove_flag(self) -> None:
        """Remove flag from configuration."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(name="test", strategy=RolloutStrategy.ALL)
        config.add_flag(flag)
        config.remove_flag("test")
        assert config.get_flag("test") is None

    def test_list_active_flags(self) -> None:
        """List only non-expired flags."""
        config = FeatureFlagConfig()
        active_flag = FeatureFlag(
            name="active",
            strategy=RolloutStrategy.ALL,
        )
        expired_flag = FeatureFlag(
            name="expired",
            strategy=RolloutStrategy.ALL,
            expires_at=datetime.now(tz=None) - timedelta(days=1),
        )
        config.add_flag(active_flag)
        config.add_flag(expired_flag)

        active = config.list_active_flags()
        assert len(active) == 1
        assert active[0].name == "active"

    def test_list_expired_flags(self) -> None:
        """List only expired flags."""
        config = FeatureFlagConfig()
        expired_flag = FeatureFlag(
            name="expired",
            strategy=RolloutStrategy.ALL,
            expires_at=datetime.now(tz=None) - timedelta(days=1),
        )
        config.add_flag(expired_flag)

        expired = config.list_expired_flags()
        assert len(expired) == 1
        assert expired[0].name == "expired"


class TestFeatureFlagEvaluator:
    """Tests for feature flag evaluation logic."""

    def test_evaluate_all_strategy(self) -> None:
        """ALL strategy enables for all users."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(name="all_flag", enabled=True, strategy=RolloutStrategy.ALL)
        config.add_flag(flag)

        evaluator = FeatureFlagEvaluator(config)
        assert evaluator.is_enabled("all_flag")
        assert evaluator.is_enabled("all_flag", user_id="user_1")
        assert evaluator.is_enabled("all_flag", user_id="user_2")

    def test_evaluate_none_strategy(self) -> None:
        """NONE strategy disables for all users."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(name="none_flag", enabled=True, strategy=RolloutStrategy.NONE)
        config.add_flag(flag)

        evaluator = FeatureFlagEvaluator(config)
        assert not evaluator.is_enabled("none_flag")
        assert not evaluator.is_enabled("none_flag", user_id="user_1")

    def test_evaluate_disabled_flag(self) -> None:
        """Disabled flag returns False regardless of strategy."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(
            name="disabled_flag",
            enabled=False,
            strategy=RolloutStrategy.ALL,
        )
        config.add_flag(flag)

        evaluator = FeatureFlagEvaluator(config)
        assert not evaluator.is_enabled("disabled_flag")

    def test_evaluate_expired_flag(self) -> None:
        """Expired flag returns False."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(
            name="expired_flag",
            enabled=True,
            strategy=RolloutStrategy.ALL,
            expires_at=datetime.now(tz=None) - timedelta(days=1),
        )
        config.add_flag(flag)

        evaluator = FeatureFlagEvaluator(config)
        assert not evaluator.is_enabled("expired_flag")

    def test_percentage_rollout_deterministic(self) -> None:
        """Same user always gets same percentage result."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(
            name="percentage_flag",
            enabled=True,
            strategy=RolloutStrategy.PERCENTAGE,
            percentage_rollout=PercentageRollout(percentage=50, seed="test"),
        )
        config.add_flag(flag)

        evaluator = FeatureFlagEvaluator(config)
        result1 = evaluator.is_enabled("percentage_flag", user_id="user_1")
        result2 = evaluator.is_enabled("percentage_flag", user_id="user_1")
        assert result1 == result2

    def test_percentage_rollout_distribution(self) -> None:
        """50% rollout affects roughly 50% of users."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(
            name="percentage_flag",
            enabled=True,
            strategy=RolloutStrategy.PERCENTAGE,
            percentage_rollout=PercentageRollout(percentage=50, seed="test"),
        )
        config.add_flag(flag)

        evaluator = FeatureFlagEvaluator(config)
        enabled_count = sum(
            1 for i in range(100) if evaluator.is_enabled("percentage_flag", user_id=f"user_{i}")
        )
        # Allow 40-60% range due to randomness in hashing
        assert 40 <= enabled_count <= 60

    def test_gradual_rollout_before_start(self) -> None:
        """Gradual rollout disabled before start date."""
        now = datetime.now(tz=None)
        config = FeatureFlagConfig()
        flag = FeatureFlag(
            name="gradual_flag",
            enabled=True,
            strategy=RolloutStrategy.GRADUAL,
            gradual_rollout=GradualRollout(
                start_date=now + timedelta(days=1),
                end_date=now + timedelta(days=2),
                percentages=[(now + timedelta(days=1), 50)],
            ),
        )
        config.add_flag(flag)

        evaluator = FeatureFlagEvaluator(config)
        assert not evaluator.is_enabled("gradual_flag")

    def test_user_targeting_include(self) -> None:
        """User targeting includes specific users."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(
            name="targeting_flag",
            enabled=True,
            strategy=RolloutStrategy.USER_TARGETING,
            user_targeting=UserTargeting(include_users=["user_1", "user_2"]),
        )
        config.add_flag(flag)

        evaluator = FeatureFlagEvaluator(config)
        assert evaluator.is_enabled("targeting_flag", user_id="user_1")
        assert evaluator.is_enabled("targeting_flag", user_id="user_2")
        assert not evaluator.is_enabled("targeting_flag", user_id="user_3")

    def test_user_targeting_exclude(self) -> None:
        """User exclusion takes precedence."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(
            name="targeting_flag",
            enabled=True,
            strategy=RolloutStrategy.USER_TARGETING,
            user_targeting=UserTargeting(
                include_users=["user_1", "user_2"],
                exclude_users=["user_2"],
            ),
        )
        config.add_flag(flag)

        evaluator = FeatureFlagEvaluator(config)
        assert evaluator.is_enabled("targeting_flag", user_id="user_1")
        assert not evaluator.is_enabled("targeting_flag", user_id="user_2")

    def test_user_targeting_segments(self) -> None:
        """Segment-based targeting."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(
            name="targeting_flag",
            enabled=True,
            strategy=RolloutStrategy.USER_TARGETING,
            user_targeting=UserTargeting(
                include_segments=["beta", "enterprise"],
            ),
        )
        config.add_flag(flag)

        evaluator = FeatureFlagEvaluator(config)
        assert evaluator.is_enabled("targeting_flag", user_segment="beta")
        assert evaluator.is_enabled("targeting_flag", user_segment="enterprise")
        assert not evaluator.is_enabled("targeting_flag", user_segment="free")


class TestFeatureFlagManager:
    """Tests for feature flag manager."""

    @pytest.mark.asyncio
    async def test_manager_is_enabled(self) -> None:
        """Manager evaluates flags correctly."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(name="test", enabled=True, strategy=RolloutStrategy.ALL)
        config.add_flag(flag)

        manager = FeatureFlagManager(config)
        assert await manager.is_enabled("test")

    @pytest.mark.asyncio
    async def test_manager_metrics_tracking(self) -> None:
        """Manager tracks evaluation metrics."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(name="test", enabled=True, strategy=RolloutStrategy.ALL)
        config.add_flag(flag)

        manager = FeatureFlagManager(config)
        await manager.is_enabled("test")
        await manager.is_enabled("test")

        metrics = manager.get_metrics("test")
        assert metrics["evaluation"] == 2
        assert metrics["enabled"] == 2

    def test_manager_update_flag(self) -> None:
        """Manager updates flag configuration."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(
            name="test",
            enabled=False,
            strategy=RolloutStrategy.ALL,
        )
        config.add_flag(flag)

        manager = FeatureFlagManager(config)
        updated_flag = FeatureFlag(
            name="test",
            enabled=True,
            strategy=RolloutStrategy.ALL,
        )
        manager.update_flag(updated_flag)

        assert config.get_flag("test").enabled

    def test_manager_remove_flag(self) -> None:
        """Manager removes flags."""
        config = FeatureFlagConfig()
        flag = FeatureFlag(name="test", enabled=True, strategy=RolloutStrategy.ALL)
        config.add_flag(flag)

        manager = FeatureFlagManager(config)
        manager.remove_flag("test")

        assert config.get_flag("test") is None

    def test_manager_cleanup_expired(self) -> None:
        """Manager cleans up expired flags."""
        config = FeatureFlagConfig()
        expired_flag = FeatureFlag(
            name="expired",
            enabled=True,
            strategy=RolloutStrategy.ALL,
            expires_at=datetime.now(tz=None) - timedelta(days=1),
        )
        config.add_flag(expired_flag)

        manager = FeatureFlagManager(config)
        count = manager.cleanup_expired_flags()

        assert count == 1
        assert config.get_flag("expired") is None
