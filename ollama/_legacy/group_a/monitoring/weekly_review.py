"""Weekly metrics review module.

Provides analysis and visualization functions for weekly metrics review.
Can be executed as a script or imported into Jupyter notebooks.

Usage:
  - As a script: python -m ollama.monitoring.weekly_review
  - In Jupyter: from ollama.monitoring.weekly_review import generate_weekly_report
"""

from typing import Any

from ollama.monitoring.metrics import AgentMetrics, MetricsCollector


def generate_weekly_report(collector: MetricsCollector) -> dict[str, Any]:
    """Generate comprehensive weekly metrics report.

    Args:
        collector: MetricsCollector instance with aggregated metrics

    Returns:
        Formatted weekly report dictionary
    """
    report = collector.get_weekly_report()
    kill_signals = collector.kill_signal_check()

    # Enhance report with analysis
    report["analysis"] = _analyze_metrics(collector)
    report["kill_signals"] = kill_signals
    report["recommendations"] = _generate_recommendations(collector)

    return report


def _analyze_metrics(collector: MetricsCollector) -> dict[str, Any]:
    """Perform analytical review of metrics."""
    analysis: dict[str, Any] = {
        "quality_trend": {},
        "performance_trend": {},
        "alerts": [],
    }

    for agent_id, metrics in collector.agent_metrics.items():
        # Quality trend
        quality_score = _calculate_quality_score(metrics)
        analysis["quality_trend"][agent_id] = {
            "agent_name": metrics.agent_name,
            "quality_score": quality_score,
            "meets_bar": metrics.meets_quality_bar(),
        }

        # Performance trend
        analysis["performance_trend"][agent_id] = {
            "agent_name": metrics.agent_name,
            "avg_latency_improvement": "TBD",  # Would compare to previous week
            "accuracy_trend": "stable",  # Would compare to previous week
        }

        # Generate alerts
        if metrics.hallucination_rate > 0.01:
            analysis["alerts"].append(
                {
                    "severity": "warning",
                    "agent_id": agent_id,
                    "message": f"Hallucination rate {metrics.hallucination_rate:.2%} approaching threshold",
                }
            )

        if metrics.p95_latency_ms > 250000:
            analysis["alerts"].append(
                {
                    "severity": "warning",
                    "agent_id": agent_id,
                    "message": f"P95 latency {metrics.p95_latency_ms:.0f}ms approaching 5min threshold",
                }
            )

    return analysis


def _calculate_quality_score(metrics: AgentMetrics) -> float:
    """Calculate overall quality score (0-100)."""
    weights = {
        "hallucination": 0.3,  # Lower hallucination = higher score
        "accuracy": 0.3,  # Higher accuracy = higher score
        "latency": 0.2,  # Lower latency = higher score
        "override": 0.2,  # Lower override rate = higher score
    }

    # Normalize metrics to 0-100 scale
    hallucination_score = max(0, 100 - (metrics.hallucination_rate * 100 * 50))
    accuracy_score = metrics.action_accuracy * 100
    latency_score = max(0, 100 - ((metrics.p95_latency_ms / 300000) * 100))
    override_score = max(0, 100 - (metrics.human_override_rate * 100))

    # Calculate weighted score
    quality_score = (
        weights["hallucination"] * hallucination_score
        + weights["accuracy"] * accuracy_score
        + weights["latency"] * latency_score
        + weights["override"] * override_score
    )

    return round(quality_score, 2)


def _generate_recommendations(collector: MetricsCollector) -> list[dict[str, Any]]:
    """Generate actionable recommendations based on metrics."""
    recommendations: list[dict[str, Any]] = []

    for agent_id, metrics in collector.agent_metrics.items():
        if metrics.hallucination_rate > 0.015:
            recommendations.append(
                {
                    "agent_id": agent_id,
                    "agent_name": metrics.agent_name,
                    "type": "training",
                    "priority": "high",
                    "action": "Refine system prompt and increase example coverage",
                    "expected_impact": "Reduce hallucination rate by 50%",
                }
            )

        if metrics.action_accuracy < 0.96:
            recommendations.append(
                {
                    "agent_id": agent_id,
                    "agent_name": metrics.agent_name,
                    "type": "training",
                    "priority": "high",
                    "action": "Add domain-specific training and adversarial scenarios",
                    "expected_impact": "Improve accuracy to >97%",
                }
            )

        if metrics.p95_latency_ms > 250000:
            recommendations.append(
                {
                    "agent_id": agent_id,
                    "agent_name": metrics.agent_name,
                    "type": "optimization",
                    "priority": "high",
                    "action": "Reduce reasoning steps or parallelize operations",
                    "expected_impact": "Reduce P95 latency by 30-40%",
                }
            )

        if metrics.human_override_rate > 0.25:
            recommendations.append(
                {
                    "agent_id": agent_id,
                    "agent_name": metrics.agent_name,
                    "type": "investigation",
                    "priority": "high",
                    "action": "Review override reasons and identify patterns",
                    "expected_impact": "Identify and fix root cause issues",
                }
            )

    return recommendations


def print_weekly_report(report: dict[str, Any]) -> None:
    """Print formatted weekly report to console.

    Args:
        report: Weekly report dictionary
    """
    print("\n" + "=" * 80)
    print("WEEKLY METRICS REPORT")
    print("=" * 80)
    print(f"\nReport Date: {report['report_date']}")
    print(f"Week Of: {report['week_of']}\n")

    # Summary
    summary = report["summary"]
    print(f"Total Agents: {summary['total_agents']}")
    print(f"Meeting Quality Bar: {summary['agents_meeting_quality_bar']}")
    print(f"Needing Attention: {len(summary['agents_needing_attention'])}\n")

    # Agent Details
    print("AGENT METRICS:")
    print("-" * 80)
    for agent in report["agents"]:
        print(f"\nAgent: {agent['agent_name']} ({agent['agent_id']})")
        print(f"  Type: {agent['agent_type']}")
        print("  Quality Metrics:")
        for key, value in agent["quality_metrics"].items():
            print(f"    {key}: {value}")
        print("  Execution Metrics:")
        for key, value in agent["execution_metrics"].items():
            print(f"    {key}: {value}")
        print(f"  Meets Quality Bar: {agent['meets_quality_bar']}")

    # Kill Signals
    if report["kill_signals"]["has_signals"]:
        print("\n" + "!" * 80)
        print("⚠️  KILL SIGNALS DETECTED")
        print("!" * 80)
        for signal in report["kill_signals"]["signals"]:
            print(f"\n  Signal: {signal['type']}")
            print(f"  Agent: {signal['agent_name']} ({signal['agent_id']})")
            print(f"  Value: {signal['value']}")
            print(f"  Action: {signal['action']}")

    # Recommendations
    if report["recommendations"]:
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        for rec in report["recommendations"]:
            print(f"\n[{rec['priority'].upper()}] {rec['agent_name']}")
            print(f"  Type: {rec['type']}")
            print(f"  Action: {rec['action']}")
            print(f"  Expected Impact: {rec['expected_impact']}")

    print("\n" + "=" * 80 + "\n")


def main() -> None:
    """Run weekly review as standalone script."""
    print("Weekly Metrics Review")
    print("This module generates weekly reports from collected metrics.")
    print("\nUsage:")
    print("  from ollama.monitoring.weekly_review import generate_weekly_report")
    print("  report = generate_weekly_report(collector)")


if __name__ == "__main__":
    main()
