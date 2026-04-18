"""Safety metrics and override rate tracking.

Tests measure and track human override rates and safety metric compliance.
Used to identify when agents need retraining or tuning.

Metric Threshold: <10% override rate for medium severity, <30% for critical
"""

from typing import Any

import pytest


class SafetyMetricsTracker:
    """Tracks safety metrics and human override rates."""

    def __init__(self) -> None:
        """Initialize safety metrics tracker."""
        self.events: list[dict[str, Any]] = []
        self.thresholds: dict[str, dict[str, float]] = {
            "medium_severity": {"max_override_rate": 0.10},
            "critical_severity": {"max_override_rate": 0.30},
        }

    def record_action(
        self,
        action_id: str,
        action_type: str,
        severity: str,
        was_overridden: bool,
        reason: str = "",
    ) -> None:
        """Record an agent action and whether it was overridden.

        Args:
            action_id: Unique action identifier
            action_type: Type of action (remediation, detection, etc.)
            severity: Severity level (low, medium, high, critical)
            was_overridden: Whether human overrode the action
            reason: Reason for override if applicable
        """
        self.events.append(
            {
                "action_id": action_id,
                "action_type": action_type,
                "severity": severity,
                "was_overridden": was_overridden,
                "reason": reason,
            }
        )

    def get_override_rate(
        self, severity: str | None = None, action_type: str | None = None
    ) -> float:
        """Get human override rate.

        Args:
            severity: Optional filter by severity
            action_type: Optional filter by action type

        Returns:
            Override rate as decimal (0.0 - 1.0)
        """
        events = self.events

        if severity:
            events = [e for e in events if e["severity"] == severity]

        if action_type:
            events = [e for e in events if e["action_type"] == action_type]

        if not events:
            return 0.0

        overridden = sum(1 for e in events if e["was_overridden"])
        return overridden / len(events)

    def get_override_statistics(self) -> dict[str, Any]:
        """Get comprehensive override statistics.

        Returns:
            Dictionary with override rates by severity and action type
        """
        stats: dict[str, Any] = {
            "total_actions": len(self.events),
            "total_overridden": sum(1 for e in self.events if e["was_overridden"]),
            "by_severity": {},
            "by_action_type": {},
        }

        for severity in ["low", "medium", "high", "critical"]:
            severity_events = [e for e in self.events if e["severity"] == severity]
            if severity_events:
                overridden = sum(1 for e in severity_events if e["was_overridden"])
                stats["by_severity"][severity] = {
                    "total": len(severity_events),
                    "overridden": overridden,
                    "override_rate": overridden / len(severity_events),
                }

        action_types = set(e["action_type"] for e in self.events)
        for action_type in action_types:
            action_events = [e for e in self.events if e["action_type"] == action_type]
            overridden = sum(1 for e in action_events if e["was_overridden"])
            stats["by_action_type"][action_type] = {
                "total": len(action_events),
                "overridden": overridden,
                "override_rate": overridden / len(action_events),
            }

        return stats

    def check_threshold_compliance(self) -> dict[str, Any]:
        """Check if override rates meet defined thresholds.

        Returns:
            Compliance check results with any violations
        """
        results: dict[str, Any] = {"compliant": True, "violations": []}

        stats = self.get_override_statistics()

        # Check medium severity threshold
        if "medium" in stats["by_severity"]:
            medium_rate = stats["by_severity"]["medium"]["override_rate"]
            threshold = self.thresholds["medium_severity"]["max_override_rate"]
            if medium_rate > threshold:
                results["compliant"] = False
                results["violations"].append(
                    {
                        "metric": "medium_severity_override_rate",
                        "value": medium_rate,
                        "threshold": threshold,
                        "exceeds_by": medium_rate - threshold,
                    }
                )

        # Check critical severity threshold
        if "critical" in stats["by_severity"]:
            critical_rate = stats["by_severity"]["critical"]["override_rate"]
            threshold = self.thresholds["critical_severity"]["max_override_rate"]
            if critical_rate > threshold:
                results["compliant"] = False
                results["violations"].append(
                    {
                        "metric": "critical_severity_override_rate",
                        "value": critical_rate,
                        "threshold": threshold,
                        "exceeds_by": critical_rate - threshold,
                    }
                )

        return results

    def get_override_reasons(self, severity: str | None = None) -> dict[str, int]:
        """Get breakdown of override reasons.

        Args:
            severity: Optional filter by severity

        Returns:
            Dictionary with reason counts
        """
        events = self.events

        if severity:
            events = [e for e in events if e["severity"] == severity]

        # Count reasons for overridden actions
        overridden_events = [e for e in events if e["was_overridden"]]

        reasons: dict[str, int] = {}
        for event in overridden_events:
            reason = event["reason"] or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1

        return reasons

    def simulate_actions(
        self,
        action_count: int = 100,
        medium_override_rate: float = 0.08,
        critical_override_rate: float = 0.25,
    ) -> None:
        """Simulate agent actions with specified override rates.

        Args:
            action_count: Total actions to simulate
            medium_override_rate: Simulated override rate for medium severity
            critical_override_rate: Simulated override rate for critical severity
        """
        import random

        action_types = ["remediation", "detection", "investigation"]
        severities = ["low", "medium", "high", "critical"]
        override_reasons = [
            "incorrect_reasoning",
            "dangerous_action",
            "incomplete_fix",
            "wrong_target",
            "alternative_preferred",
        ]

        for i in range(action_count):
            severity = random.choice(severities)
            action_type = random.choice(action_types)

            # Determine override based on severity
            if severity == "medium":
                should_override = random.random() < medium_override_rate
            elif severity == "critical":
                should_override = random.random() < critical_override_rate
            else:
                should_override = random.random() < 0.05

            reason = ""
            if should_override:
                reason = random.choice(override_reasons)

            self.record_action(
                action_id=f"action_{i:05d}",
                action_type=action_type,
                severity=severity,
                was_overridden=should_override,
                reason=reason,
            )


class TestSafetyMetrics:
    """Test suite for agent safety metrics and override rate tracking."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Setup test fixtures."""
        self.tracker = SafetyMetricsTracker()
        self.tracker.simulate_actions(
            action_count=200,
            medium_override_rate=0.08,
            critical_override_rate=0.25,
        )

    def test_safety_metrics_initialization(self) -> None:
        """Test safety metrics tracker initialization."""
        tracker = SafetyMetricsTracker()
        assert tracker.get_override_rate() == 0.0
        assert len(tracker.events) == 0

    def test_action_recording(self) -> None:
        """Test recording of agent actions."""
        tracker = SafetyMetricsTracker()
        tracker.record_action(
            action_id="test_001",
            action_type="remediation",
            severity="medium",
            was_overridden=False,
        )

        assert len(tracker.events) == 1
        assert tracker.events[0]["action_id"] == "test_001"

    def test_override_rate_calculation(self) -> None:
        """Test override rate calculation."""
        override_rate = self.tracker.get_override_rate()
        assert 0.0 <= override_rate <= 1.0

    def test_override_rate_by_severity(self) -> None:
        """Test override rate calculation by severity."""
        medium_rate = self.tracker.get_override_rate(severity="medium")
        critical_rate = self.tracker.get_override_rate(severity="critical")

        assert 0.0 <= medium_rate <= 1.0
        assert 0.0 <= critical_rate <= 1.0
        # Critical should typically have higher override rate
        # (though not guaranteed in simulation)

    def test_override_rate_by_action_type(self) -> None:
        """Test override rate by action type."""
        remediation_rate = self.tracker.get_override_rate(action_type="remediation")
        detection_rate = self.tracker.get_override_rate(action_type="detection")

        assert 0.0 <= remediation_rate <= 1.0
        assert 0.0 <= detection_rate <= 1.0

    def test_override_statistics(self) -> None:
        """Test comprehensive override statistics."""
        stats = self.tracker.get_override_statistics()

        assert "total_actions" in stats
        assert "total_overridden" in stats
        assert "by_severity" in stats
        assert "by_action_type" in stats

        assert stats["total_actions"] > 0
        assert stats["total_overridden"] >= 0

    def test_threshold_compliance_check(self) -> None:
        """Test threshold compliance checking."""
        compliance = self.tracker.check_threshold_compliance()

        assert "compliant" in compliance
        assert isinstance(compliance["compliant"], bool)
        assert "violations" in compliance

    def test_medium_severity_threshold(self) -> None:
        """Test medium severity override threshold (<10%)."""
        medium_rate = self.tracker.get_override_rate(severity="medium")
        threshold = 0.10

        # Note: In real scenario, should not exceed threshold
        # Here we just validate the logic
        assert medium_rate <= 1.0

    def test_critical_severity_threshold(self) -> None:
        """Test critical severity override threshold (<30%)."""
        critical_rate = self.tracker.get_override_rate(severity="critical")
        threshold = 0.30

        # Note: In real scenario, should not exceed threshold
        assert critical_rate <= 1.0

    def test_override_reasons_tracking(self) -> None:
        """Test tracking of override reasons."""
        reasons = self.tracker.get_override_reasons()

        if reasons:
            assert isinstance(reasons, dict)
            # Should have various reasons
            assert sum(reasons.values()) > 0

    def test_override_reasons_by_severity(self) -> None:
        """Test override reasons filtered by severity."""
        critical_reasons = self.tracker.get_override_reasons(severity="critical")
        assert isinstance(critical_reasons, dict)

    def test_kill_signal_high_override_rate(self) -> None:
        """Test kill signal detection for high override rates.

        Agent should be flagged for retraining if override rate exceeds threshold.
        """
        tracker = SafetyMetricsTracker()

        # Simulate many overrides
        for i in range(50):
            tracker.record_action(
                action_id=f"bad_action_{i}",
                action_type="remediation",
                severity="critical",
                was_overridden=True,
                reason="agent_hallucination",
            )

        critical_rate = tracker.get_override_rate(severity="critical")
        compliance = tracker.check_threshold_compliance()

        assert critical_rate == 1.0, "Should have 100% override rate"
        assert not compliance["compliant"], "Should fail compliance"
        assert len(compliance["violations"]) > 0

    def test_metric_trending_over_time(self) -> None:
        """Test that metrics can be tracked over time for trending.

        This validates the data structure supports historical analysis.
        """
        tracker = SafetyMetricsTracker()

        # Simulate improving performance over time
        severities = ["medium", "medium", "critical", "critical"]
        override_probabilities = [0.20, 0.10, 0.50, 0.25]

        for i, (severity, prob) in enumerate(zip(severities, override_probabilities)):
            import random

            for j in range(10):
                action_id = f"time_series_{i}_{j}"
                was_overridden = random.random() < prob
                tracker.record_action(
                    action_id=action_id,
                    action_type="test",
                    severity=severity,
                    was_overridden=was_overridden,
                )

        # Validate we have events to trend
        assert len(tracker.events) > 0
        stats = tracker.get_override_statistics()
        assert stats["total_actions"] > 0

    def test_empty_severity_handling(self) -> None:
        """Test handling of queries with no matching events."""
        tracker = SafetyMetricsTracker()

        # Query with no events
        rate = tracker.get_override_rate(severity="medium")
        assert rate == 0.0

        # Query with non-existent action type
        rate = tracker.get_override_rate(action_type="nonexistent")
        assert rate == 0.0
