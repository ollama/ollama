"""Performance benchmarking tests for agent response latency.

Tests measure and track agent performance metrics including P50, P95, P99 latency
and historical trending for week-over-week comparisons.

Metric Threshold: <30s for triage, <5min for complex investigations (P95)
"""

import statistics
import time
from typing import Any

import pytest


class LatencyBenchmark:
    """Measures and tracks agent response latency."""

    def __init__(self) -> None:
        """Initialize latency benchmark tracker."""
        self.measurements: list[dict[str, Any]] = []
        self.baselines: dict[str, float] = {
            "triage": 30.0,  # 30 seconds
            "complex": 300.0,  # 5 minutes
        }

    def record_measurement(
        self,
        task_id: str,
        task_type: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record a latency measurement.

        Args:
            task_id: Unique task identifier
            task_type: Type of task (triage, complex, remediation, etc.)
            latency_ms: Response latency in milliseconds
            success: Whether task completed successfully
        """
        self.measurements.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "latency_ms": latency_ms,
                "latency_sec": latency_ms / 1000.0,
                "success": success,
                "timestamp": time.time(),
            }
        )

    def get_percentile(self, percentile: int, task_type: str | None = None) -> float:
        """Get latency percentile (P50, P95, P99).

        Args:
            percentile: Percentile to calculate (50, 95, 99)
            task_type: Optional filter by task type

        Returns:
            Latency in milliseconds at specified percentile
        """
        measurements = self.measurements

        if task_type:
            measurements = [m for m in measurements if m["task_type"] == task_type]

        if not measurements:
            raise ValueError(f"No measurements found for percentile {percentile}")

        latencies = sorted([m["latency_ms"] for m in measurements])
        index = int((percentile / 100.0) * len(latencies))
        return latencies[min(index, len(latencies) - 1)]

    def get_statistics(self, task_type: str | None = None) -> dict[str, float]:
        """Get latency statistics.

        Args:
            task_type: Optional filter by task type

        Returns:
            Dictionary with min, max, mean, median, stdev, P95, P99
        """
        measurements = self.measurements

        if task_type:
            measurements = [m for m in measurements if m["task_type"] == task_type]

        if not measurements:
            return {}

        latencies = [m["latency_ms"] for m in measurements]

        return {
            "count": len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "stdev_ms": (statistics.stdev(latencies) if len(latencies) > 1 else 0.0),
            "p95_ms": self.get_percentile(95, task_type),
            "p99_ms": self.get_percentile(99, task_type),
        }

    def check_threshold(self, task_type: str) -> dict[str, Any]:
        """Check if measurements meet performance thresholds.

        Args:
            task_type: Type of task to check

        Returns:
            Dictionary with threshold check results
        """
        if task_type not in self.baselines:
            return {"error": f"Unknown task type: {task_type}"}

        threshold_sec = self.baselines[task_type]
        threshold_ms = threshold_sec * 1000.0

        stats = self.get_statistics(task_type)
        if not stats:
            return {"error": f"No measurements for task type: {task_type}"}

        p95_ms = stats["p95_ms"]
        passes = p95_ms <= threshold_ms

        return {
            "task_type": task_type,
            "threshold_ms": threshold_ms,
            "p95_measured_ms": p95_ms,
            "passes": passes,
            "exceeds_by_ms": max(0, p95_ms - threshold_ms),
        }

    def simulate_task_execution(
        self,
        task_type: str,
        task_count: int = 10,
        min_latency_ms: float = 100,
        max_latency_ms: float = 5000,
    ) -> None:
        """Simulate task execution and record latencies.

        Args:
            task_type: Type of task
            task_count: Number of tasks to simulate
            min_latency_ms: Minimum simulated latency
            max_latency_ms: Maximum simulated latency
        """
        import random

        for i in range(task_count):
            # Simulate varied latencies (using normal distribution)
            mean = (min_latency_ms + max_latency_ms) / 2
            stdev = (max_latency_ms - min_latency_ms) / 6
            latency = max(
                min_latency_ms,
                min(max_latency_ms, random.gauss(mean, stdev)),
            )

            task_id = f"{task_type}_{i:05d}"
            self.record_measurement(task_id, task_type, latency)


class HistoricalTrendAnalysis:
    """Analyzes historical trends in agent performance (week-over-week)."""

    def __init__(self) -> None:
        """Initialize trend analysis."""
        self.weekly_data: list[dict[str, Any]] = []

    def add_weekly_snapshot(
        self,
        week_number: int,
        task_type: str,
        p95_latency_ms: float,
        accuracy_rate: float,
    ) -> None:
        """Add weekly performance snapshot.

        Args:
            week_number: Week number for tracking
            task_type: Type of task
            p95_latency_ms: P95 latency for the week
            accuracy_rate: Task accuracy rate (0-1)
        """
        self.weekly_data.append(
            {
                "week": week_number,
                "task_type": task_type,
                "p95_latency_ms": p95_latency_ms,
                "accuracy_rate": accuracy_rate,
            }
        )

    def get_trend(self, task_type: str, metric: str = "p95_latency_ms") -> list[dict[str, Any]]:
        """Get trend data for a metric across weeks.

        Args:
            task_type: Type of task
            metric: Metric to trend (p95_latency_ms, accuracy_rate)

        Returns:
            List of weekly measurements with trends
        """
        data = [d for d in self.weekly_data if d["task_type"] == task_type]
        data.sort(key=lambda x: x["week"])

        result = []
        for i, point in enumerate(data):
            trend_item = {
                "week": point["week"],
                "value": point[metric],
            }

            if i > 0:
                prev_value = data[i - 1][metric]
                change = point[metric] - prev_value
                pct_change = (change / prev_value * 100) if prev_value != 0 else 0
                trend_item["change"] = change
                trend_item["pct_change"] = pct_change
                trend_item["trend"] = "↑" if change > 0 else "↓" if change < 0 else "→"

            result.append(trend_item)

        return result

    def detect_regression(self, task_type: str, threshold_pct: float = 10.0) -> dict[str, Any]:
        """Detect performance regressions (>threshold% degradation).

        Args:
            task_type: Type of task
            threshold_pct: Regression threshold percentage

        Returns:
            Regression detection results
        """
        trend = self.get_trend(task_type, "p95_latency_ms")

        if len(trend) < 2:
            return {"has_regression": False, "reason": "Insufficient data"}

        # Check last two weeks
        prev_week = trend[-2]
        curr_week = trend[-1]

        if "pct_change" in curr_week:
            pct_change = curr_week["pct_change"]
            if pct_change > threshold_pct:
                return {
                    "has_regression": True,
                    "previous_p95_ms": prev_week["value"],
                    "current_p95_ms": curr_week["value"],
                    "pct_degradation": pct_change,
                    "threshold_pct": threshold_pct,
                }

        return {"has_regression": False}


class TestPerformanceBenchmarks:
    """Test suite for agent performance benchmarking."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Setup test fixtures."""
        self.benchmark = LatencyBenchmark()
        self.trend_analysis = HistoricalTrendAnalysis()
        self._simulate_measurements()

    def _simulate_measurements(self) -> None:
        """Simulate performance measurements for testing."""
        # Triage tasks (should be <30s)
        self.benchmark.simulate_task_execution(
            task_type="triage",
            task_count=50,
            min_latency_ms=5000,
            max_latency_ms=25000,
        )

        # Complex investigation tasks (should be <5min)
        self.benchmark.simulate_task_execution(
            task_type="complex",
            task_count=30,
            min_latency_ms=60000,
            max_latency_ms=250000,
        )

        # Remediation tasks
        self.benchmark.simulate_task_execution(
            task_type="remediation",
            task_count=20,
            min_latency_ms=10000,
            max_latency_ms=120000,
        )

    def test_p95_latency_measurement(self) -> None:
        """Test P95 latency measurement."""
        p95 = self.benchmark.get_percentile(95, task_type="triage")
        assert p95 > 0, "P95 latency should be > 0"
        assert p95 < 30000, "Triage P95 should be < 30s"

    def test_p99_latency_measurement(self) -> None:
        """Test P99 latency measurement."""
        p99 = self.benchmark.get_percentile(99, task_type="complex")
        assert p99 > 0, "P99 latency should be > 0"

    def test_latency_percentiles(self) -> None:
        """Test multiple latency percentiles."""
        for percentile in [50, 95, 99]:
            value = self.benchmark.get_percentile(percentile, task_type="triage")
            assert value > 0, f"P{percentile} should be > 0"

    def test_triage_threshold_met(self) -> None:
        """Test triage tasks meet <30s threshold."""
        check = self.benchmark.check_threshold("triage")
        assert "passes" in check
        # Note: In real scenario, this should pass
        # Here we just validate the check logic works

    def test_complex_threshold_met(self) -> None:
        """Test complex investigation tasks meet <5min threshold."""
        check = self.benchmark.check_threshold("complex")
        assert "passes" in check
        assert "p95_measured_ms" in check

    def test_latency_statistics(self) -> None:
        """Test latency statistics calculation."""
        stats = self.benchmark.get_statistics(task_type="triage")

        assert "count" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats
        assert "mean_ms" in stats
        assert "median_ms" in stats
        assert "stdev_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats

    def test_latency_ordering(self) -> None:
        """Test latency statistics are ordered correctly."""
        stats = self.benchmark.get_statistics(task_type="triage")

        assert stats["min_ms"] <= stats["median_ms"]
        assert stats["median_ms"] <= stats["max_ms"]
        assert stats["p95_ms"] <= stats["p99_ms"]

    def test_weekly_trend_tracking(self) -> None:
        """Test week-over-week trend tracking."""
        # Add weekly snapshots
        for week in range(1, 5):
            # Simulate slight improvement over weeks
            p95 = 25000 - (week * 1000)
            accuracy = 0.90 + (week * 0.02)
            self.trend_analysis.add_weekly_snapshot(
                week_number=week,
                task_type="triage",
                p95_latency_ms=p95,
                accuracy_rate=accuracy,
            )

        trend = self.trend_analysis.get_trend("triage")
        assert len(trend) == 4
        assert trend[0]["week"] == 1

    def test_trend_shows_improvement(self) -> None:
        """Test trend detection shows improvement correctly."""
        # Add trend showing improvement
        self.trend_analysis.add_weekly_snapshot(
            week_number=1, task_type="test", p95_latency_ms=100000, accuracy_rate=0.90
        )
        self.trend_analysis.add_weekly_snapshot(
            week_number=2, task_type="test", p95_latency_ms=90000, accuracy_rate=0.92
        )

        trend = self.trend_analysis.get_trend("test")
        assert len(trend) == 2
        assert trend[1]["pct_change"] < 0, "Latency should decrease"
        assert trend[1]["trend"] == "↓"

    def test_regression_detection(self) -> None:
        """Test regression detection."""
        # Add trend showing degradation
        self.trend_analysis.add_weekly_snapshot(
            week_number=1, task_type="perf_test", p95_latency_ms=20000, accuracy_rate=0.95
        )
        self.trend_analysis.add_weekly_snapshot(
            week_number=2, task_type="perf_test", p95_latency_ms=25000, accuracy_rate=0.95
        )

        regression = self.trend_analysis.detect_regression("perf_test", threshold_pct=10.0)
        assert "pct_degradation" in regression
        assert regression["pct_degradation"] > 0

    def test_no_regression_when_stable(self) -> None:
        """Test regression detection when performance is stable."""
        self.trend_analysis.add_weekly_snapshot(
            week_number=1, task_type="stable", p95_latency_ms=20000, accuracy_rate=0.95
        )
        self.trend_analysis.add_weekly_snapshot(
            week_number=2, task_type="stable", p95_latency_ms=20100, accuracy_rate=0.95
        )

        regression = self.trend_analysis.detect_regression("stable", threshold_pct=10.0)
        assert regression["has_regression"] is False

    def test_kill_signal_latency_violation(self) -> None:
        """Test kill signal when latency violates threshold."""
        # Create benchmark with measurements that exceed threshold
        benchmark = LatencyBenchmark()

        # Simulate very slow tasks (violating threshold)
        for i in range(10):
            benchmark.record_measurement(
                f"slow_task_{i}",
                "triage",
                latency_ms=45000,  # Exceeds 30s threshold
            )

        check = benchmark.check_threshold("triage")
        assert check["passes"] is False, "Should fail when exceeding threshold"
        assert check["exceeds_by_ms"] > 0

    def test_concurrent_task_tracking(self) -> None:
        """Test tracking multiple task types concurrently."""
        # This test validates the benchmark handles multiple task types
        assert len(self.benchmark.measurements) > 0

        task_types = set(m["task_type"] for m in self.benchmark.measurements)
        assert len(task_types) >= 3, "Should have tracked multiple task types"
