"""
Chaos Experiment Manager
========================

Manages lifecycle of chaos experiments: scheduling, execution, monitoring,
and rollback.

Provides:
    - Experiment scheduling (immediate, delayed, canary-triggered)
    - Concurrent experiment execution (with max limits)
    - Real-time monitoring and health checks
    - Automatic rollback on failure
    - Metrics collection and analysis
    - Structured logging with experiment context

Example:
    >>> from ollama.services.chaos import ChaosManager, ChaosExperiment
    >>> manager = ChaosManager()
    >>> exp = manager.get_experiment("inference_latency_spike")
    >>> experiment_id = manager.schedule_experiment(
    ...     experiment=exp,
    ...     delay_seconds=0,
    ...     run_in_background=True
    ... )
    >>> # Monitor experiment
    >>> status = manager.get_experiment_status(experiment_id)
    >>> print(f"Status: {status.state}, Error rate: {status.error_rate}")
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from ollama.services.chaos.config import (
    ChaosConfig,
    ChaosExperiment,
    get_chaos_config,
)

log = structlog.get_logger(__name__)


class ExperimentState(str, Enum):
    """State of chaos experiment."""

    PENDING = "pending"  # Scheduled, awaiting start
    RUNNING = "running"  # Currently executing
    PAUSED = "paused"  # Temporarily paused
    COMPLETED = "completed"  # Successfully completed
    ROLLED_BACK = "rolled_back"  # Rolled back
    FAILED = "failed"  # Failed during execution


@dataclass
class ExperimentResult:
    """Result of chaos experiment execution."""

    experiment_id: str
    experiment_name: str
    state: ExperimentState
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float | None = None
    error_rate: float = 0.0  # Percentage of failed requests
    error_count: int = 0
    success_count: int = 0
    timeout_count: int = 0
    cascading_failures: int = 0
    circuit_breaker_trips: int = 0
    observed_failure_modes: list[str] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    rollback_reason: str | None = None

    @property
    def total_requests(self) -> int:
        """Calculate total requests."""
        return self.success_count + self.error_count + self.timeout_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.success_count / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "state": self.state,
            "duration_seconds": self.duration_seconds,
            "error_rate": self.error_rate,
            "error_count": self.error_count,
            "success_count": self.success_count,
            "timeout_count": self.timeout_count,
            "cascading_failures": self.cascading_failures,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "total_requests": self.total_requests,
            "success_rate": self.success_rate,
            "observed_failure_modes": self.observed_failure_modes,
        }


class ChaosManager:
    """Manages chaos experiment lifecycle."""

    def __init__(self, config: ChaosConfig | None = None) -> None:
        """Initialize chaos manager.

        Args:
            config: Chaos configuration (uses default if not provided)
        """
        self.config = config or get_chaos_config()
        self.running_experiments: dict[str, ExperimentResult] = {}
        self.experiment_history: list[ExperimentResult] = []
        self.max_concurrent = self.config.max_concurrent_experiments

    def schedule_experiment(
        self,
        experiment: ChaosExperiment,
        delay_seconds: int = 0,
        run_in_background: bool = False,
    ) -> str:
        """Schedule a chaos experiment.

        Args:
            experiment: Experiment to schedule
            delay_seconds: Delay before starting (0 = immediate)
            run_in_background: Run asynchronously

        Returns:
            Experiment ID

        Raises:
            ValueError: If max concurrent experiments exceeded or chaos disabled
        """
        if not self.config.enabled:
            raise ValueError("Chaos engineering is disabled")

        if len(self.running_experiments) >= self.max_concurrent and not run_in_background:
            raise ValueError(f"Max concurrent experiments ({self.max_concurrent}) reached")

        experiment_id = str(uuid.uuid4())

        result = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=experiment.name,
            state=ExperimentState.PENDING,
            start_time=datetime.now(),
        )

        log.info(
            "experiment_scheduled",
            experiment_id=experiment_id,
            experiment_name=experiment.name,
            delay_seconds=delay_seconds,
            run_in_background=run_in_background,
        )

        if run_in_background:
            asyncio.create_task(self._execute_experiment(result, experiment, delay_seconds))
        else:
            # Synchronous execution
            asyncio.run(self._execute_experiment(result, experiment, delay_seconds))

        self.running_experiments[experiment_id] = result
        return experiment_id

    async def _execute_experiment(
        self,
        result: ExperimentResult,
        experiment: ChaosExperiment,
        delay_seconds: int = 0,
    ) -> None:
        """Execute chaos experiment.

        Args:
            result: Result object to populate
            experiment: Experiment to execute
            delay_seconds: Initial delay
        """
        try:
            # Apply initial delay
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

            # Update state to running
            result.state = ExperimentState.RUNNING
            log.info(
                "experiment_started",
                experiment_id=result.experiment_id,
                experiment_name=experiment.name,
            )

            # Execute experiment for specified duration
            start_time = datetime.now()
            end_time = start_time + timedelta(seconds=experiment.duration_seconds)

            # Simulate experiment execution
            while datetime.now() < end_time:
                # Simulate chaos impact
                await self._simulate_chaos_impact(result, experiment)

                # Check health
                if experiment.health_check.abort_on_unhealthy:
                    if result.error_rate > 0.5:
                        result.state = ExperimentState.ROLLED_BACK
                        result.rollback_reason = (
                            f"Error rate exceeded threshold: {result.error_rate}"
                        )
                        log.warning(
                            "experiment_rolled_back",
                            experiment_id=result.experiment_id,
                            reason=result.rollback_reason,
                        )
                        return

                # Check rollback threshold
                if experiment.rollback_config.on_threshold_breach:
                    if result.error_rate > experiment.rollback_config.error_rate_threshold:
                        result.state = ExperimentState.ROLLED_BACK
                        result.rollback_reason = (
                            f"Error rate {result.error_rate} exceeded "
                            f"threshold {experiment.rollback_config.error_rate_threshold}"
                        )
                        log.warning(
                            "experiment_rolled_back_threshold",
                            experiment_id=result.experiment_id,
                            error_rate=result.error_rate,
                            threshold=experiment.rollback_config.error_rate_threshold,
                        )
                        return

                # Wait before next check
                await asyncio.sleep(1)

            # Mark as completed
            result.state = ExperimentState.COMPLETED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()

            log.info(
                "experiment_completed",
                experiment_id=result.experiment_id,
                duration_seconds=result.duration_seconds,
                error_rate=result.error_rate,
                cascading_failures=result.cascading_failures,
            )

            # Add to history
            self.experiment_history.append(result)

            # Remove from running
            self.running_experiments.pop(result.experiment_id, None)

        except Exception as e:
            result.state = ExperimentState.FAILED
            result.end_time = datetime.now()
            log.error(
                "experiment_failed",
                experiment_id=result.experiment_id,
                error=str(e),
            )
            self.running_experiments.pop(result.experiment_id, None)

    async def _simulate_chaos_impact(
        self,
        result: ExperimentResult,
        experiment: ChaosExperiment,
    ) -> None:
        """Simulate chaos experiment impact on metrics.

        Args:
            result: Result object to update
            experiment: Experiment being simulated
        """
        # Simulate request distribution
        total_requests = 100

        # Calculate failure rate based on severity
        severity_map = {
            "low": 0.05,
            "medium": 0.15,
            "high": 0.35,
            "critical": 0.60,
        }
        expected_error_rate = severity_map.get(experiment.severity.value, 0.15)

        # Add randomness
        import random

        actual_error_rate = expected_error_rate + random.uniform(-0.05, 0.05)
        actual_error_rate = max(0, min(1.0, actual_error_rate))

        # Update counts
        errors = int(total_requests * actual_error_rate)
        timeouts = int(errors * 0.3)  # 30% of errors are timeouts
        successes = total_requests - errors

        result.error_count += errors
        result.timeout_count += timeouts
        result.success_count += successes
        result.error_rate = (result.error_count / max(result.total_requests, 1)) * 100

        # Simulate failure modes
        if random.random() < 0.1:
            result.circuit_breaker_trips += 1
        if random.random() < 0.05:
            result.cascading_failures += 1

    def get_experiment_status(self, experiment_id: str) -> ExperimentResult | None:
        """Get current status of experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment result or None
        """
        return self.running_experiments.get(experiment_id)

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a running experiment.

        Args:
            experiment_id: Experiment to stop

        Returns:
            True if stopped, False if not found
        """
        result = self.running_experiments.get(experiment_id)
        if result and result.state == ExperimentState.RUNNING:
            result.state = ExperimentState.PAUSED
            result.end_time = datetime.now()
            log.info("experiment_stopped", experiment_id=experiment_id)
            return True
        return False

    def rollback_experiment(self, experiment_id: str, reason: str = "Manual rollback") -> bool:
        """Manually rollback an experiment.

        Args:
            experiment_id: Experiment to rollback
            reason: Rollback reason

        Returns:
            True if rolled back, False if not found or not running
        """
        result = self.running_experiments.get(experiment_id)
        if result and result.state in [
            ExperimentState.RUNNING,
            ExperimentState.PAUSED,
        ]:
            result.state = ExperimentState.ROLLED_BACK
            result.rollback_reason = reason
            log.info(
                "experiment_rolled_back_manual",
                experiment_id=experiment_id,
                reason=reason,
            )
            return True
        return False

    def get_experiment_history(
        self,
        limit: int = 100,
        experiment_name: str | None = None,
    ) -> list[ExperimentResult]:
        """Get experiment history.

        Args:
            limit: Maximum results to return
            experiment_name: Filter by experiment name

        Returns:
            List of experiment results
        """
        history = self.experiment_history

        if experiment_name:
            history = [e for e in history if e.experiment_name == experiment_name]

        return sorted(history, key=lambda x: x.start_time, reverse=True)[:limit]

    def get_running_experiments(self) -> list[ExperimentResult]:
        """Get all running experiments.

        Returns:
            List of running experiment results
        """
        return list(self.running_experiments.values())

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary metrics across all experiments.

        Returns:
            Metrics summary dictionary
        """
        history = self.experiment_history

        if not history:
            return {
                "total_experiments": 0,
                "avg_error_rate": 0.0,
                "cascading_failures": 0,
                "circuit_breaker_trips": 0,
            }

        avg_error_rate = sum(e.error_rate for e in history) / len(history)
        total_cascading = sum(e.cascading_failures for e in history)
        total_cb_trips = sum(e.circuit_breaker_trips for e in history)

        return {
            "total_experiments": len(history),
            "avg_error_rate": avg_error_rate,
            "cascading_failures": total_cascading,
            "circuit_breaker_trips": total_cb_trips,
            "recovery_time_avg_seconds": sum(e.duration_seconds or 0 for e in history)
            / len(history),
        }
