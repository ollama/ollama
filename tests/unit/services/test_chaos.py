"""
Tests for Chaos Engineering Module
===================================

Tests chaos experiment configuration, management, execution, and metrics.
"""

from datetime import datetime, timedelta

import pytest

from ollama.services.chaos import (
    ChaosExperiment,
    ChaosManager,
    ChaosMetrics,
    ExperimentState,
    ExperimentType,
    SeverityLevel,
)
from ollama.services.chaos.config import (
    ComputeConfig,
    FailureMode,
    NetworkConfig,
    get_chaos_config,
)


class TestChaosExperimentCreation:
    """Test chaos experiment creation and configuration."""

    def test_create_network_latency_experiment(self) -> None:
        """Network latency experiment can be created."""
        exp = ChaosExperiment(
            name="test_latency",
            experiment_type=ExperimentType.NETWORK_LATENCY,
            duration_seconds=300,
            severity=SeverityLevel.MEDIUM,
            target_service="inference",
        )

        assert exp.name == "test_latency"
        assert exp.experiment_type == ExperimentType.NETWORK_LATENCY
        assert exp.duration_seconds == 300
        assert exp.severity == SeverityLevel.MEDIUM
        assert exp.target_service == "inference"

    def test_network_config_defaults(self) -> None:
        """Network config has sensible defaults."""
        config = NetworkConfig()

        assert config.latency_ms == 0
        assert config.jitter_ms == 0
        assert config.packet_loss_percent == 0
        assert config.bandwidth_limit_mbps == 0

    def test_compute_config_with_values(self) -> None:
        """Compute config accepts custom values."""
        config = ComputeConfig(
            cpu_throttle_percent=50,
            memory_limit_mb=512,
            duration_seconds=60,
        )

        assert config.cpu_throttle_percent == 50
        assert config.memory_limit_mb == 512
        assert config.duration_seconds == 60

    def test_default_chaos_config(self) -> None:
        """Default chaos config includes sample experiments."""
        config = get_chaos_config()

        assert config.enabled is True
        assert len(config.experiments) >= 5
        assert any(e.name == "inference_latency_spike" for e in config.experiments)

    def test_experiment_type_enum(self) -> None:
        """All experiment types are defined."""
        valid_types = [
            ExperimentType.NETWORK_LATENCY,
            ExperimentType.NETWORK_LOSS,
            ExperimentType.SERVICE_FAILURE,
            ExperimentType.CASCADING_FAILURE,
            ExperimentType.RESOURCE_CPU,
            ExperimentType.RESOURCE_MEMORY,
            ExperimentType.STATE_INCONSISTENCY,
            ExperimentType.DEPENDENCY_LATENCY,
        ]

        assert len(valid_types) == 8
        for exp_type in valid_types:
            assert isinstance(exp_type, ExperimentType)

    def test_severity_levels(self) -> None:
        """Severity levels are correctly defined."""
        levels = [
            SeverityLevel.LOW,
            SeverityLevel.MEDIUM,
            SeverityLevel.HIGH,
            SeverityLevel.CRITICAL,
        ]

        assert len(levels) == 4
        for level in levels:
            assert isinstance(level, SeverityLevel)

    def test_failure_modes(self) -> None:
        """Failure modes are correctly defined."""
        modes = [
            FailureMode.GRACEFUL_DEGRADATION,
            FailureMode.CIRCUIT_BREAKER_TRIP,
            FailureMode.TIMEOUT,
            FailureMode.CASCADING_PROPAGATION,
            FailureMode.RESOURCE_EXHAUSTION,
            FailureMode.STATE_INCONSISTENCY,
            FailureMode.SILENT_FAILURE,
        ]

        assert len(modes) == 7
        for mode in modes:
            assert isinstance(mode, FailureMode)


class TestChaosManager:
    """Test chaos experiment manager."""

    def test_manager_initialization(self) -> None:
        """Manager initializes with configuration."""
        manager = ChaosManager()

        assert manager.config is not None
        assert manager.max_concurrent == 3
        assert len(manager.running_experiments) == 0
        assert len(manager.experiment_history) == 0

    def test_schedule_experiment_immediate(self) -> None:
        """Experiment can be scheduled for immediate execution."""
        manager = ChaosManager()
        config = get_chaos_config()
        exp = config.experiments[0]

        experiment_id = manager.schedule_experiment(
            experiment=exp,
            delay_seconds=0,
            run_in_background=True,
        )

        assert experiment_id is not None
        assert len(experiment_id) == 36  # UUID format

    def test_schedule_experiment_with_delay(self) -> None:
        """Experiment can be scheduled with delay."""
        manager = ChaosManager()
        config = get_chaos_config()
        exp = config.experiments[0]

        experiment_id = manager.schedule_experiment(
            experiment=exp,
            delay_seconds=5,
            run_in_background=True,
        )

        status = manager.get_experiment_status(experiment_id)
        assert status is not None
        assert status.state == ExperimentState.PENDING

    def test_max_concurrent_experiments(self) -> None:
        """Max concurrent experiments limit is enforced."""
        manager = ChaosManager()
        manager.max_concurrent = 1

        config = get_chaos_config()
        exp1 = config.experiments[0]
        exp2 = config.experiments[1]

        # First experiment succeeds
        id1 = manager.schedule_experiment(
            experiment=exp1,
            run_in_background=True,
        )
        assert id1 is not None

        # Second should raise error (max concurrent = 1)
        with pytest.raises(ValueError, match="Max concurrent"):
            manager.schedule_experiment(
                experiment=exp2,
                run_in_background=False,
            )

    def test_get_experiment_history(self) -> None:
        """Experiment history can be retrieved."""
        manager = ChaosManager()

        # Add dummy result to history
        from ollama.services.chaos.manager import ExperimentResult

        result = ExperimentResult(
            experiment_id="test-123",
            experiment_name="test_exp",
            state=ExperimentState.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=60),
            duration_seconds=60.0,
            error_rate=5.0,
        )
        manager.experiment_history.append(result)

        history = manager.get_experiment_history(limit=10)
        assert len(history) == 1
        assert history[0].experiment_name == "test_exp"

    def test_metrics_summary(self) -> None:
        """Metrics summary can be computed."""
        manager = ChaosManager()

        summary = manager.get_metrics_summary()

        assert "total_experiments" in summary
        assert "avg_error_rate" in summary
        assert "cascading_failures" in summary
        assert "circuit_breaker_trips" in summary

    def test_stop_experiment(self) -> None:
        """Running experiment can be stopped."""
        manager = ChaosManager()
        config = get_chaos_config()
        exp = config.experiments[0]

        experiment_id = manager.schedule_experiment(
            experiment=exp,
            run_in_background=True,
        )

        status = manager.get_experiment_status(experiment_id)
        assert status is not None

        # Simulate running state
        from ollama.services.chaos.manager import ExperimentResult

        manager.running_experiments[experiment_id] = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=exp.name,
            state=ExperimentState.RUNNING,
            start_time=datetime.now(),
        )

        stopped = manager.stop_experiment(experiment_id)
        assert stopped is True

    def test_rollback_experiment_manual(self) -> None:
        """Experiment can be manually rolled back."""
        manager = ChaosManager()
        from ollama.services.chaos.manager import ExperimentResult

        result = ExperimentResult(
            experiment_id="test-456",
            experiment_name="test_exp",
            state=ExperimentState.RUNNING,
            start_time=datetime.now(),
        )
        manager.running_experiments["test-456"] = result

        rolled_back = manager.rollback_experiment("test-456", "Manual stop")
        assert rolled_back is True
        assert result.state == ExperimentState.ROLLED_BACK


class TestChaosMetrics:
    """Test chaos metrics collection."""

    def test_metrics_initialization(self) -> None:
        """Metrics collector initializes."""
        metrics = ChaosMetrics()

        assert metrics.registry is not None
        assert len(metrics.experiments) == 0

    def test_record_chaos_started(self) -> None:
        """Chaos start can be recorded."""
        metrics = ChaosMetrics()

        metrics.record_chaos_started("exp-1", "test_experiment")

        assert "exp-1" in metrics.experiments
        assert metrics.experiments["exp-1"].experiment_name == "test_experiment"

    def test_record_request_succeeded(self) -> None:
        """Successful request can be recorded."""
        metrics = ChaosMetrics()
        metrics.record_chaos_started("exp-1", "test_exp")

        metrics.record_request_succeeded("exp-1", latency_ms=100.0)

        exp_metrics = metrics.get_experiment_metrics("exp-1")
        assert exp_metrics.requests_total == 1
        assert exp_metrics.requests_succeeded == 1
        assert exp_metrics.success_rate == 1.0

    def test_record_request_failed(self) -> None:
        """Failed request can be recorded."""
        metrics = ChaosMetrics()
        metrics.record_chaos_started("exp-1", "test_exp")

        metrics.record_request_failed("exp-1", error_type="timeout", latency_ms=5000.0)

        exp_metrics = metrics.get_experiment_metrics("exp-1")
        assert exp_metrics.requests_total == 1
        assert exp_metrics.requests_timeout == 1
        assert exp_metrics.timeout_rate == 1.0

    def test_record_circuit_breaker_trip(self) -> None:
        """Circuit breaker trip can be recorded."""
        metrics = ChaosMetrics()
        metrics.record_chaos_started("exp-1", "test_exp")

        metrics.record_circuit_breaker_trip("exp-1")
        metrics.record_circuit_breaker_trip("exp-1")

        exp_metrics = metrics.get_experiment_metrics("exp-1")
        assert exp_metrics.circuit_breaker_trips == 2

    def test_record_cascading_failure(self) -> None:
        """Cascading failure can be recorded."""
        metrics = ChaosMetrics()
        metrics.record_chaos_started("exp-1", "test_exp")

        metrics.record_cascading_failure("exp-1")

        exp_metrics = metrics.get_experiment_metrics("exp-1")
        assert exp_metrics.cascading_failures == 1

    def test_record_failure_mode(self) -> None:
        """Failure mode observation can be recorded."""
        metrics = ChaosMetrics()
        metrics.record_chaos_started("exp-1", "test_exp")

        metrics.record_failure_mode("exp-1", "graceful_degradation")
        metrics.record_failure_mode("exp-1", "timeout")

        exp_metrics = metrics.get_experiment_metrics("exp-1")
        assert len(exp_metrics.observed_failure_modes) == 2
        assert "graceful_degradation" in exp_metrics.observed_failure_modes

    def test_record_chaos_completed(self) -> None:
        """Chaos completion with metrics can be recorded."""
        metrics = ChaosMetrics()
        metrics.record_chaos_started("exp-1", "test_exp")

        # Add some requests
        for i in range(100):
            metrics.record_request_succeeded("exp-1", latency_ms=100.0 + i)

        metrics.record_chaos_completed("exp-1", recovery_time_seconds=30.0)

        exp_metrics = metrics.get_experiment_metrics("exp-1")
        assert exp_metrics.recovery_time_seconds == 30.0
        assert exp_metrics.latency_p50_ms > 0
        assert exp_metrics.latency_p99_ms > 0

    def test_aggregate_metrics(self) -> None:
        """Aggregate metrics can be computed."""
        metrics = ChaosMetrics()

        # Create first experiment
        metrics.record_chaos_started("exp-1", "test_exp_1")
        metrics.record_request_succeeded("exp-1", latency_ms=100.0)
        metrics.record_chaos_completed("exp-1", recovery_time_seconds=30.0)

        # Create second experiment
        metrics.record_chaos_started("exp-2", "test_exp_2")
        metrics.record_request_succeeded("exp-2", latency_ms=200.0)
        metrics.record_request_failed("exp-2", error_type="timeout", latency_ms=5000.0)
        metrics.record_chaos_completed("exp-2", recovery_time_seconds=60.0)

        aggregate = metrics.get_aggregate_metrics()

        assert aggregate["total_experiments"] == 2
        assert aggregate["avg_success_rate"] > 0
        assert aggregate["avg_recovery_time_seconds"] > 0


class TestChaosExperimentIntegration:
    """Integration tests for chaos experiments."""

    def test_end_to_end_chaos_experiment(self) -> None:
        """End-to-end chaos experiment workflow."""
        manager = ChaosManager()
        metrics = ChaosMetrics()

        config = get_chaos_config()
        exp = config.experiments[0]

        # Schedule experiment
        experiment_id = manager.schedule_experiment(
            experiment=exp,
            run_in_background=True,
        )

        assert experiment_id is not None

        # Record metrics
        metrics.record_chaos_started(experiment_id, exp.name)
        metrics.record_request_succeeded(experiment_id, latency_ms=150.0)
        metrics.record_request_failed(experiment_id, error_type="timeout", latency_ms=5000.0)
        metrics.record_circuit_breaker_trip(experiment_id)
        metrics.record_chaos_completed(experiment_id, recovery_time_seconds=45.0)

        # Verify metrics
        exp_metrics = metrics.get_experiment_metrics(experiment_id)
        assert exp_metrics is not None
        assert exp_metrics.recovery_time_seconds == 45.0

    def test_multiple_concurrent_experiments(self) -> None:
        """Multiple experiments can run concurrently."""
        manager = ChaosManager()
        manager.max_concurrent = 5

        config = get_chaos_config()

        experiment_ids = []
        for i, exp in enumerate(config.experiments[:3]):
            exp_id = manager.schedule_experiment(
                experiment=exp,
                run_in_background=True,
            )
            experiment_ids.append(exp_id)

        assert len(experiment_ids) == 3
        running = manager.get_running_experiments()
        assert len(running) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
