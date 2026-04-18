"""
Cost Management Service - Issue #46 Phase 1

High-level service for cost tracking, forecasting, and optimization.
Integrates GCP Cost Collector with FastAPI endpoints and database persistence.

Version: 1.0.0 (Week 1 - Feb 3, 2026)
Status: PRODUCTION-READY
"""

from datetime import UTC, datetime
from typing import Any

import structlog

from ollama.services.cost.collector import CostCategory, CostSnapshot, GCPCostCollector

log = structlog.get_logger(__name__)


class CostManagementService:
    """
    High-level cost management service.

    Provides:
    - Real-time cost tracking
    - Monthly budget tracking
    - Cost forecasting
    - Alerts and anomaly detection
    - Cost optimization recommendations
    - Multi-project aggregation
    """

    def __init__(
        self,
        project_id: str,
        billing_account_id: str,
        monthly_budget_usd: float = 50000.0
    ):
        """
        Initialize Cost Management Service.

        Args:
            project_id: Primary GCP project
            billing_account_id: GCP billing account
            monthly_budget_usd: Monthly spending budget
        """
        self.project_id = project_id
        self.billing_account_id = billing_account_id
        self.monthly_budget_usd = monthly_budget_usd

        # Initialize collector
        self.collector = GCPCostCollector(
            project_id=project_id,
            billing_account_id=billing_account_id
        )

        # Tracking
        self.current_month_cost = 0.0
        self.alerts: list[dict[str, Any]] = []

        log.info(
            "cost_service_initialized",
            project_id=project_id,
            monthly_budget_usd=monthly_budget_usd
        )

    async def collect_and_update(self) -> CostSnapshot:
        """
        Collect costs and update tracking.

        Returns:
            Latest CostSnapshot
        """
        try:
            snapshot = await self.collector.collect_costs()

            # Update monthly total
            current_month = datetime.now(UTC).strftime("%Y-%m")
            self.current_month_cost = sum(
                s.total_cost_usd
                for s in self.collector.hourly_snapshots
                if s.timestamp.strftime("%Y-%m") == current_month
            )

            # Check alerts
            self._update_alerts(snapshot)

            return snapshot

        except Exception as e:
            log.error("cost_update_failed", error=str(e))
            raise

    def _update_alerts(self, snapshot: CostSnapshot) -> None:
        """Generate alerts based on current costs."""
        self.alerts = []

        # Budget alert
        budget_percent = (self.current_month_cost / self.monthly_budget_usd) * 100
        if budget_percent > 80:
            severity = "critical" if budget_percent > 100 else "warning"
            self.alerts.append({
                "type": "budget_alert",
                "severity": severity,
                "message": f"Budget usage at {budget_percent:.1f}%",
                "budget_usd": self.monthly_budget_usd,
                "current_usd": self.current_month_cost,
                "percent": budget_percent
            })

        # Anomaly alerts
        for anomaly in snapshot.anomalies_detected:
            self.alerts.append({
                "type": "anomaly_alert",
                "severity": anomaly.get("severity", "medium"),
                "message": anomaly.get("message"),
                "details": anomaly
            })

    def get_cost_summary(self) -> dict[str, Any]:
        """
        Get comprehensive cost summary.

        Returns:
            Cost summary with current, forecast, budget info
        """
        estimated_monthly = self.collector.estimate_monthly_cost()
        budget_percent = (self.current_month_cost / self.monthly_budget_usd) * 100

        return {
            "current_month_cost_usd": round(self.current_month_cost, 2),
            "estimated_monthly_cost_usd": round(estimated_monthly, 2),
            "monthly_budget_usd": self.monthly_budget_usd,
            "budget_percent": round(budget_percent, 1),
            "remaining_budget_usd": round(self.monthly_budget_usd - self.current_month_cost, 2),
            "forecast_status": "on_track" if estimated_monthly <= self.monthly_budget_usd else "over_budget",
            "days_in_month": 30,  # Approximation
            "current_daily_average": round(self.current_month_cost / max(1, datetime.now(UTC).day), 2),
            "projected_daily_average": round(estimated_monthly / 30, 2),
            "alerts_count": len(self.alerts),
            "alerts": self.alerts
        }

    def get_cost_breakdown(self) -> dict[str, Any]:
        """
        Get detailed cost breakdown.

        Returns:
            Breakdown by category, service, project, region
        """
        if not self.collector.hourly_snapshots:
            return {}

        # Get latest snapshot
        latest = self.collector.hourly_snapshots[-1]

        return {
            "by_category": {
                k.value: round(v, 2)
                for k, v in latest.cost_by_category.items()
            },
            "by_service": {
                k: round(v, 2)
                for k, v in latest.cost_by_service.items()
            },
            "by_project": {
                k: round(v, 2)
                for k, v in latest.cost_by_project.items()
            },
            "by_region": {
                k: round(v, 2)
                for k, v in latest.cost_by_region.items()
            },
            "timestamp": latest.timestamp.isoformat()
        }

    def get_cost_trend(self, hours: int = 24) -> dict[str, Any]:
        """
        Get cost trend data.

        Args:
            hours: Number of hours to include

        Returns:
            Trend data with timestamps and costs
        """
        trend = self.collector.get_cost_trend(hours)

        return {
            "period_hours": hours,
            "data_points": [
                {
                    "timestamp": ts.isoformat(),
                    "cost_usd": round(cost, 2)
                }
                for ts, cost in trend
            ],
            "min_cost_usd": round(min((c for _, c in trend), default=0), 2),
            "max_cost_usd": round(max((c for _, c in trend), default=0), 2),
            "avg_cost_usd": round(sum(c for _, c in trend) / len(trend) if trend else 0, 2)
        }

    def get_optimization_recommendations(self) -> list[dict[str, Any]]:
        """
        Get cost optimization recommendations.

        Returns:
            List of actionable recommendations
        """
        recommendations: list[dict[str, Any]] = []

        if not self.collector.hourly_snapshots:
            return recommendations

        latest = self.collector.hourly_snapshots[-1]

        # Recommendation 1: High Compute costs
        if latest.cost_by_category.get(CostCategory.COMPUTE, 0) > 3000:
            recommendations.append({
                "category": "compute",
                "priority": "high",
                "title": "Optimize Compute Costs",
                "description": "Consider using committed use discounts or reducing cluster size",
                "potential_savings_percent": 15,
                "action": "Review GKE node pool sizing and enable autoscaling"
            })

        # Recommendation 2: Storage optimization
        if latest.cost_by_category.get(CostCategory.STORAGE, 0) > 400:
            recommendations.append({
                "category": "storage",
                "priority": "medium",
                "title": "Optimize Storage",
                "description": "Enable lifecycle policies for old objects",
                "potential_savings_percent": 20,
                "action": "Set up Cloud Storage lifecycle policies for 90-day archive"
            })

        # Recommendation 3: Network optimization
        if latest.cost_by_category.get(CostCategory.NETWORK, 0) > 200:
            recommendations.append({
                "category": "network",
                "priority": "medium",
                "title": "Reduce Network Egress",
                "description": "Use Cloud CDN to cache content and reduce egress",
                "potential_savings_percent": 30,
                "action": "Enable Cloud CDN on Load Balancer"
            })

        # Recommendation 4: Database optimization
        if latest.cost_by_category.get(CostCategory.DATABASE, 0) > 500:
            recommendations.append({
                "category": "database",
                "priority": "medium",
                "title": "Optimize Database",
                "description": "Consider managed databases vs self-hosted",
                "potential_savings_percent": 25,
                "action": "Review Cloud SQL instance sizing and reserved capacity"
            })

        return recommendations

    async def generate_cost_report(self) -> dict[str, Any]:
        """
        Generate comprehensive cost report.

        Returns:
            Full cost report with all metrics
        """
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": self.get_cost_summary(),
            "breakdown": self.get_cost_breakdown(),
            "trend": self.get_cost_trend(hours=24),
            "recommendations": self.get_optimization_recommendations(),
            "alerts": self.alerts
        }
