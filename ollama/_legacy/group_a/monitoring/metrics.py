"""Metrics collection and aggregation module.

Collects agent performance metrics, system metrics, and business metrics
for weekly review and compliance tracking.

Implements Elite Execution Protocol Section: "Velocity & Quality Metrics"
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class AgentMetric:
    """Single agent performance metric data point."""

    agent_id: str
    metric_name: str
    value: float
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "agent_id": self.agent_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


@dataclass
class AgentMetrics:
    """Aggregated metrics for a single agent."""

    agent_id: str
    agent_name: str
    agent_type: str

    # Quality metrics
    hallucination_rate: float = 0.0
    action_accuracy: float = 0.0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    human_override_rate: float = 0.0

    # Execution metrics
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0

    # Compliance metrics
    violations: int = 0
    audit_log_entries: int = 0

    # Timestamp
    measurement_date: datetime = field(default_factory=datetime.now)

    def meets_quality_bar(self) -> bool:
        """Check if agent meets minimum quality standards."""
        return (
            self.hallucination_rate < 0.02  # <2%
            and self.action_accuracy > 0.95  # >95%
            and self.p95_latency_ms < 300000  # <5min
            and self.human_override_rate < 0.30  # <30%
        )

    def get_report(self) -> dict[str, Any]:
        """Get agent metrics report."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "quality_metrics": {
                "hallucination_rate": f"{self.hallucination_rate:.2%}",
                "action_accuracy": f"{self.action_accuracy:.2%}",
                "average_latency_ms": f"{self.average_latency_ms:.0f}",
                "p95_latency_ms": f"{self.p95_latency_ms:.0f}",
                "p99_latency_ms": f"{self.p99_latency_ms:.0f}",
                "human_override_rate": f"{self.human_override_rate:.2%}",
            },
            "execution_metrics": {
                "total_actions": self.total_actions,
                "successful_actions": self.successful_actions,
                "failed_actions": self.failed_actions,
                "success_rate": f"{self.successful_actions / max(1, self.total_actions):.2%}",
            },
            "compliance_metrics": {
                "violations": self.violations,
                "audit_log_entries": self.audit_log_entries,
            },
            "meets_quality_bar": self.meets_quality_bar(),
            "measurement_date": self.measurement_date.isoformat(),
        }


class MetricsCollector:
    """Collects metrics from agents and systems."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.metrics: list[AgentMetric] = []
        self.agent_metrics: dict[str, AgentMetrics] = {}

    def record_metric(
        self,
        agent_id: str,
        metric_name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record a single metric data point.

        Args:
            agent_id: Agent identifier
            metric_name: Name of metric
            value: Metric value
            tags: Optional tags for filtering
        """
        metric = AgentMetric(
            agent_id=agent_id,
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
        )
        self.metrics.append(metric)

    def aggregate_metrics(  # noqa: C901
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
    ) -> AgentMetrics:
        """Aggregate metrics for an agent.

        Args:
            agent_id: Agent identifier
            agent_name: Human-readable agent name
            agent_type: Type of agent (security, monitoring, etc.)

        Returns:
            Aggregated metrics
        """
        agent_metrics_data = AgentMetrics(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
        )

        # Filter metrics for this agent
        agent_metrics_list = [m for m in self.metrics if m.agent_id == agent_id]

        if not agent_metrics_list:
            return agent_metrics_data

        # Aggregate by metric name
        metric_groups: dict[str, list[float]] = {}
        for metric in agent_metrics_list:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric.value)

        # Set aggregated values
        if "hallucination_rate" in metric_groups:
            agent_metrics_data.hallucination_rate = sum(metric_groups["hallucination_rate"]) / len(
                metric_groups["hallucination_rate"]
            )

        if "action_accuracy" in metric_groups:
            agent_metrics_data.action_accuracy = sum(metric_groups["action_accuracy"]) / len(
                metric_groups["action_accuracy"]
            )

        if "latency_ms" in metric_groups:
            latencies = metric_groups["latency_ms"]
            agent_metrics_data.average_latency_ms = sum(latencies) / len(latencies)
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            agent_metrics_data.p95_latency_ms = sorted_latencies[
                min(p95_idx, len(sorted_latencies) - 1)
            ]
            agent_metrics_data.p99_latency_ms = sorted_latencies[
                min(p99_idx, len(sorted_latencies) - 1)
            ]

        if "human_override_rate" in metric_groups:
            agent_metrics_data.human_override_rate = sum(
                metric_groups["human_override_rate"]
            ) / len(metric_groups["human_override_rate"])

        if "total_actions" in metric_groups:
            agent_metrics_data.total_actions = int(metric_groups["total_actions"][0])

        if "successful_actions" in metric_groups:
            agent_metrics_data.successful_actions = int(metric_groups["successful_actions"][0])

        if "failed_actions" in metric_groups:
            agent_metrics_data.failed_actions = int(metric_groups["failed_actions"][0])

        if "violations" in metric_groups:
            agent_metrics_data.violations = int(metric_groups["violations"][0])

        if "audit_log_entries" in metric_groups:
            agent_metrics_data.audit_log_entries = int(metric_groups["audit_log_entries"][0])

        self.agent_metrics[agent_id] = agent_metrics_data
        return agent_metrics_data

    def get_weekly_report(self) -> dict[str, Any]:
        """Generate weekly metrics report.

        Returns:
            Weekly report with all metrics and compliance status
        """
        report: dict[str, Any] = {
            "report_date": datetime.now().isoformat(),
            "week_of": (datetime.now() - timedelta(days=datetime.now().weekday())).isoformat(),
            "agents": [],
            "summary": {
                "total_agents": len(self.agent_metrics),
                "agents_meeting_quality_bar": 0,
                "agents_needing_attention": [],
            },
        }

        for agent_id, metrics in self.agent_metrics.items():
            agent_report = metrics.get_report()
            report["agents"].append(agent_report)

            if metrics.meets_quality_bar():
                report["summary"]["agents_meeting_quality_bar"] += 1
            else:
                report["summary"]["agents_needing_attention"].append(
                    {
                        "agent_id": agent_id,
                        "agent_name": metrics.agent_name,
                        "reason": self._get_failure_reason(metrics),
                    }
                )

        return report

    def _get_failure_reason(self, metrics: AgentMetrics) -> str:
        """Get reason agent doesn't meet quality bar."""
        reasons = []

        if metrics.hallucination_rate >= 0.02:
            reasons.append(f"hallucination {metrics.hallucination_rate:.2%} >= 2%")

        if metrics.action_accuracy <= 0.95:
            reasons.append(f"accuracy {metrics.action_accuracy:.2%} <= 95%")

        if metrics.p95_latency_ms >= 300000:
            reasons.append(f"P95 latency {metrics.p95_latency_ms:.0f}ms >= 5min")

        if metrics.human_override_rate >= 0.30:
            reasons.append(f"override rate {metrics.human_override_rate:.2%} >= 30%")

        return "; ".join(reasons) if reasons else "unknown"

    def export_to_json(self, filename: str) -> None:
        """Export metrics to JSON file.

        Args:
            filename: Output JSON filename
        """
        report = self.get_weekly_report()
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

    def kill_signal_check(self) -> dict[str, Any]:
        """Check for kill signals requiring escalation.

        Returns:
            Dictionary with any detected kill signals
        """
        kill_signals: dict[str, Any] = {
            "has_signals": False,
            "signals": [],
        }

        for agent_id, metrics in self.agent_metrics.items():
            if metrics.hallucination_rate >= 0.02:
                kill_signals["has_signals"] = True
                kill_signals["signals"].append(
                    {
                        "type": "high_hallucination",
                        "agent_id": agent_id,
                        "agent_name": metrics.agent_name,
                        "value": metrics.hallucination_rate,
                        "action": "Archive agent and redesign after 2 weeks",
                    }
                )

            if metrics.action_accuracy <= 0.95:
                kill_signals["has_signals"] = True
                kill_signals["signals"].append(
                    {
                        "type": "low_accuracy",
                        "agent_id": agent_id,
                        "agent_name": metrics.agent_name,
                        "value": metrics.action_accuracy,
                        "action": "Retrain or archive after 2 weeks",
                    }
                )

            if metrics.p95_latency_ms > 300000:
                kill_signals["has_signals"] = True
                kill_signals["signals"].append(
                    {
                        "type": "high_latency",
                        "agent_id": agent_id,
                        "agent_name": metrics.agent_name,
                        "value": metrics.p95_latency_ms,
                        "action": "Investigate bottleneck immediately",
                    }
                )

            if metrics.human_override_rate > 0.30:
                kill_signals["has_signals"] = True
                kill_signals["signals"].append(
                    {
                        "type": "high_override_rate",
                        "agent_id": agent_id,
                        "agent_name": metrics.agent_name,
                        "value": metrics.human_override_rate,
                        "action": "Retrain or archive agent",
                    }
                )

        return kill_signals
