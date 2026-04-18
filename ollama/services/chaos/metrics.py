"""
Chaos Experiment Metrics
========================

Collects, aggregates, and exports metrics from chaos experiments.

Provides:
    - Prometheus metrics export (failure modes, recovery times, latency)
    - Structured logging with experiment context
    - Real-time metrics aggregation
    - Historical metrics tracking
    - SLO/SLI computation
    - System health monitoring during chaos

Example:
    >>> from ollama.services.chaos import ChaosMetrics
    >>> metrics = ChaosMetrics()
    >>> metrics.record_chaos_started(experiment_id="exp-123")
    >>> metrics.record_request_failed(
    ...     experiment_id="exp-123",
    ...     error_type="timeout",
    ...     latency_ms=5000
    ... )
    >>> metrics.record_circuit_breaker_trip(experiment_id="exp-123")
    >>> prometheus_output = metrics.collect_prometheus_metrics()
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import structlog
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

log = structlog.get_logger(__name__)


@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment run."""

    experiment_id: str
    experiment_name: str
    start_time: datetime = field(default_factory=datetime.now)
    requests_total: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    requests_timeout: int = 0
    error_types: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    circuit_breaker_trips: int = 0
    cascading_failures: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_max_ms: float = 0.0
    recovery_time_seconds: float | None = None
    observed_failure_modes: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate request success rate."""
        if self.requests_total == 0:
            return 0.0
        return self.requests_succeeded / self.requests_total

    @property
    def error_rate(self) -> float:
        """Calculate request error rate."""
        if self.requests_total == 0:
            return 0.0
        return self.requests_failed / self.requests_total

    @property
    def timeout_rate(self) -> float:
        """Calculate request timeout rate."""
        if self.requests_total == 0:
            return 0.0
        return self.requests_timeout / self.requests_total


class ChaosMetrics:
    """Collects metrics from chaos experiments."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize metrics collector.

        Args:
            registry: Prometheus collector registry
        """
        self.registry = registry or CollectorRegistry()
        self.experiments: dict[str, ExperimentMetrics] = {}
        self.latencies: dict[str, list[float]] = defaultdict(list)

        # Prometheus metrics
        self.experiments_total = Counter(
            "chaos_experiments_total",
            "Total chaos experiments",
            ["status"],
            registry=self.registry,
        )

        self.experiment_duration_seconds = Histogram(
            "chaos_experiment_duration_seconds",
            "Chaos experiment duration",
            buckets=[1, 5, 10, 30, 60, 300],
            registry=self.registry,
        )

        self.requests_total = Counter(
            "chaos_requests_total",
            "Total requests during chaos",
            ["experiment_id", "status"],
            registry=self.registry,
        )

        self.request_latency_ms = Histogram(
            "chaos_request_latency_ms",
            "Request latency during chaos",
            ["experiment_id"],
            buckets=[10, 50, 100, 500, 1000, 5000],
            registry=self.registry,
        )

        self.errors_total = Counter(
            "chaos_errors_total",
            "Total errors during chaos",
            ["experiment_id", "error_type"],
            registry=self.registry,
        )

        self.circuit_breaker_trips = Counter(
            "chaos_circuit_breaker_trips_total",
            "Circuit breaker trips during chaos",
            ["experiment_id"],
            registry=self.registry,
        )

        self.cascading_failures = Counter(
            "chaos_cascading_failures_total",
            "Cascading failures detected",
            ["experiment_id"],
            registry=self.registry,
        )

        self.recovery_time_seconds = Gauge(
            "chaos_recovery_time_seconds",
            "Time to recover from chaos",
            ["experiment_id"],
            registry=self.registry,
        )

    def record_chaos_started(self, experiment_id: str, experiment_name: str) -> None:
        """Record chaos experiment start.

        Args:
            experiment_id: Unique experiment ID
            experiment_name: Name of experiment
        """
        self.experiments[experiment_id] = ExperimentMetrics(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            start_time=datetime.now(),
        )

        log.info(
            "chaos_metrics_started",
            experiment_id=experiment_id,
            experiment_name=experiment_name,
        )

    def record_request_succeeded(
        self,
        experiment_id: str,
        latency_ms: float,
    ) -> None:
        """Record successful request.

        Args:
            experiment_id: Experiment ID
            latency_ms: Request latency in milliseconds
        """
        if experiment_id not in self.experiments:
            return

        metrics = self.experiments[experiment_id]
        metrics.requests_total += 1
        metrics.requests_succeeded += 1
        self.latencies[experiment_id].append(latency_ms)

        self.requests_total.labels(experiment_id=experiment_id, status="success").inc()
        self.request_latency_ms.labels(experiment_id=experiment_id).observe(latency_ms)

    def record_request_failed(
        self,
        experiment_id: str,
        error_type: str,
        latency_ms: float = 0.0,
    ) -> None:
        """Record failed request.

        Args:
            experiment_id: Experiment ID
            error_type: Type of error (timeout, connection_error, etc.)
            latency_ms: Request latency in milliseconds
        """
        if experiment_id not in self.experiments:
            return

        metrics = self.experiments[experiment_id]
        metrics.requests_total += 1

        if error_type == "timeout":
            metrics.requests_timeout += 1
        else:
            metrics.requests_failed += 1

        metrics.error_types[error_type] += 1
        self.latencies[experiment_id].append(latency_ms)

        self.requests_total.labels(experiment_id=experiment_id, status="failed").inc()
        self.errors_total.labels(experiment_id=experiment_id, error_type=error_type).inc()

        if latency_ms > 0:
            self.request_latency_ms.labels(experiment_id=experiment_id).observe(latency_ms)

    def record_circuit_breaker_trip(self, experiment_id: str) -> None:
        """Record circuit breaker trip.

        Args:
            experiment_id: Experiment ID
        """
        if experiment_id not in self.experiments:
            return

        self.experiments[experiment_id].circuit_breaker_trips += 1
        self.circuit_breaker_trips.labels(experiment_id=experiment_id).inc()

        log.warning(
            "circuit_breaker_trip_recorded",
            experiment_id=experiment_id,
            total_trips=self.experiments[experiment_id].circuit_breaker_trips,
        )

    def record_cascading_failure(self, experiment_id: str) -> None:
        """Record cascading failure detected.

        Args:
            experiment_id: Experiment ID
        """
        if experiment_id not in self.experiments:
            return

        self.experiments[experiment_id].cascading_failures += 1
        self.cascading_failures.labels(experiment_id=experiment_id).inc()

        log.error(
            "cascading_failure_recorded",
            experiment_id=experiment_id,
            total_cascading=self.experiments[experiment_id].cascading_failures,
        )

    def record_failure_mode(self, experiment_id: str, failure_mode: str) -> None:
        """Record observed failure mode.

        Args:
            experiment_id: Experiment ID
            failure_mode: Name of failure mode
        """
        if experiment_id not in self.experiments:
            return

        metrics = self.experiments[experiment_id]
        if failure_mode not in metrics.observed_failure_modes:
            metrics.observed_failure_modes.append(failure_mode)

        log.info(
            "failure_mode_observed",
            experiment_id=experiment_id,
            failure_mode=failure_mode,
            total_modes=len(metrics.observed_failure_modes),
        )

    def record_chaos_completed(self, experiment_id: str, recovery_time_seconds: float) -> None:
        """Record chaos experiment completion.

        Args:
            experiment_id: Experiment ID
            recovery_time_seconds: Time to fully recover in seconds
        """
        if experiment_id not in self.experiments:
            return

        metrics = self.experiments[experiment_id]
        metrics.recovery_time_seconds = recovery_time_seconds

        # Calculate latency percentiles
        if self.latencies[experiment_id]:
            latencies = sorted(self.latencies[experiment_id])
            n = len(latencies)
            metrics.latency_p50_ms = latencies[int(n * 0.50)]
            metrics.latency_p95_ms = latencies[int(n * 0.95)]
            metrics.latency_p99_ms = latencies[int(n * 0.99)]
            metrics.latency_max_ms = latencies[-1]

        self.experiments_total.labels(status="completed").inc()
        self.experiment_duration_seconds.observe(recovery_time_seconds)
        self.recovery_time_seconds.labels(experiment_id=experiment_id).set(recovery_time_seconds)

        log.info(
            "chaos_metrics_completed",
            experiment_id=experiment_id,
            recovery_time_seconds=recovery_time_seconds,
            success_rate=metrics.success_rate,
            error_rate=metrics.error_rate,
            p99_latency_ms=metrics.latency_p99_ms,
            cascading_failures=metrics.cascading_failures,
            circuit_breaker_trips=metrics.circuit_breaker_trips,
            observed_failure_modes=metrics.observed_failure_modes,
        )

    def get_experiment_metrics(self, experiment_id: str) -> ExperimentMetrics | None:
        """Get metrics for specific experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment metrics or None
        """
        return self.experiments.get(experiment_id)

    def get_aggregate_metrics(self) -> dict[str, float]:
        """Get aggregate metrics across all experiments.

        Returns:
            Aggregated metrics dictionary
        """
        if not self.experiments:
            return {
                "total_experiments": 0,
                "avg_success_rate": 0.0,
                "avg_error_rate": 0.0,
                "avg_recovery_time_seconds": 0.0,
            }

        experiments = list(self.experiments.values())
        total = len(experiments)

        avg_success_rate = sum(e.success_rate for e in experiments) / total
        avg_error_rate = sum(e.error_rate for e in experiments) / total
        recovery_times = [e.recovery_time_seconds for e in experiments if e.recovery_time_seconds]
        avg_recovery = sum(recovery_times) / len(recovery_times) if recovery_times else 0.0

        total_cb_trips = sum(e.circuit_breaker_trips for e in experiments)
        total_cascading = sum(e.cascading_failures for e in experiments)

        return {
            "total_experiments": total,
            "avg_success_rate": avg_success_rate,
            "avg_error_rate": avg_error_rate,
            "avg_recovery_time_seconds": avg_recovery,
            "circuit_breaker_trips": total_cb_trips,
            "cascading_failures": total_cascading,
            "unique_failure_modes": len(
                {mode for e in experiments for mode in e.observed_failure_modes}
            ),
        }

    def collect_prometheus_metrics(self) -> str:
        """Collect all metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        from prometheus_client.exposition import generate_latest

        metrics_output = generate_latest(self.registry)
        return metrics_output.decode("utf-8")  # type: ignore[no-any-return]
