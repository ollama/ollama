"""
Unit Tests for Cost Management Service - Issue #46

Tests for GCP cost collection, aggregation, anomaly detection, and recommendations.

Status: PRODUCTION-READY
Coverage: 20+ tests for cost collection and management
Target: ✅ 100% pass rate for Week 1
"""

import pytest
from datetime import datetime, timedelta, UTC
from ollama.services.cost.collector import (
    GCPCostCollector,
    CostSample,
    CostSnapshot,
    CostCategory,
    ResourceMetric,
)
from ollama.services.cost.service import CostManagementService


@pytest.fixture
def cost_collector():
    """Create cost collector for testing."""
    return GCPCostCollector(
        project_id="test-ollama-prod",
        billing_account_id="test-billing-12345",
        region="us-central1"
    )


@pytest.fixture
def cost_service():
    """Create cost management service for testing."""
    return CostManagementService(
        project_id="test-ollama-prod",
        billing_account_id="test-billing-12345",
        monthly_budget_usd=50000.0
    )


@pytest.fixture
def sample_cost_sample():
    """Create sample cost data."""
    return CostSample(
        project_id="test-project",
        service="Compute Engine",
        category=CostCategory.COMPUTE,
        cost_usd=1250.50,
        usage_amount=500,
        usage_unit="core-hours",
        region="us-central1"
    )


class TestGCPCostCollector:
    """Tests for GCP Cost Collector."""

    def test_collector_initialization(self, cost_collector):
        """Cost collector initializes with correct parameters."""
        assert cost_collector.project_id == "test-ollama-prod"
        assert cost_collector.billing_account_id == "test-billing-12345"
        assert cost_collector.region == "us-central1"
        assert cost_collector.update_interval_seconds == 300
        assert cost_collector.retention_days == 90

    @pytest.mark.asyncio
    async def test_collect_costs(self, cost_collector):
        """Cost collection returns valid snapshot."""
        snapshot = await cost_collector.collect_costs()

        assert isinstance(snapshot, CostSnapshot)
        assert snapshot.total_cost_usd > 0
        assert len(snapshot.cost_by_category) > 0
        assert len(snapshot.cost_by_service) > 0
        assert len(snapshot.cost_by_project) > 0
        assert len(snapshot.cost_by_region) > 0

    @pytest.mark.asyncio
    async def test_cost_aggregation(self, cost_collector):
        """Costs aggregate correctly by category."""
        snapshot = await cost_collector.collect_costs()

        # Verify aggregation
        total_by_category = sum(snapshot.cost_by_category.values())
        assert abs(total_by_category - snapshot.total_cost_usd) < 0.01

    @pytest.mark.asyncio
    async def test_cost_by_service(self, cost_collector):
        """Costs aggregate correctly by service."""
        snapshot = await cost_collector.collect_costs()

        # Should have multiple services
        assert "Compute Engine" in snapshot.cost_by_service
        assert "Cloud Storage" in snapshot.cost_by_service
        assert snapshot.cost_by_service["Compute Engine"] > 0

    @pytest.mark.asyncio
    async def test_cost_by_region(self, cost_collector):
        """Costs aggregate correctly by region."""
        snapshot = await cost_collector.collect_costs()

        # Should have regions
        assert "us-central1" in snapshot.cost_by_region
        assert snapshot.cost_by_region["us-central1"] > 0

    @pytest.mark.asyncio
    async def test_anomaly_detection_enabled(self, cost_collector):
        """Anomaly detection is included in snapshots."""
        # Collect multiple times to have history
        for _ in range(2):
            await cost_collector.collect_costs()

        snapshot = await cost_collector.collect_costs()

        # Check that anomaly detection ran
        assert isinstance(snapshot.anomalies_detected, list)

    def test_cost_trend_calculation(self, cost_collector):
        """Cost trend returns correct data."""
        # Simulate some history
        now = datetime.now(UTC)
        for i in range(24):
            snapshot = CostSnapshot(
                timestamp=now - timedelta(hours=24-i),
                total_cost_usd=1000 + i*50,  # Increasing trend
                total_forecast_usd=1050 + i*50,
                cost_by_category={},
                cost_by_service={},
                cost_by_project={},
                cost_by_region={},
                anomalies_detected=[],
                data_freshness_minutes=0,
                collection_duration_seconds=1.0
            )
            cost_collector.hourly_snapshots.append(snapshot)

        trend = cost_collector.get_cost_trend(hours=24)

        assert len(trend) == 24
        assert trend[0][1] < trend[-1][1]  # Increasing trend

    def test_monthly_cost_estimation(self, cost_collector):
        """Monthly cost estimation works correctly."""
        # Simulate 24 hours of data
        now = datetime.now(UTC)
        for i in range(24):
            snapshot = CostSnapshot(
                timestamp=now - timedelta(hours=24-i),
                total_cost_usd=100.0,
                total_forecast_usd=100.0,
                cost_by_category={},
                cost_by_service={},
                cost_by_project={},
                cost_by_region={},
                anomalies_detected=[],
                data_freshness_minutes=0,
                collection_duration_seconds=1.0
            )
            cost_collector.hourly_snapshots.append(snapshot)

        estimated = cost_collector.estimate_monthly_cost()

        # Should be roughly 24 hours * 30 days = 100 * 30 = 3000
        assert 2900 < estimated < 3100

    def test_cost_sample_to_dict(self, sample_cost_sample):
        """Cost sample converts to dictionary correctly."""
        data = sample_cost_sample.to_dict()

        assert data["project_id"] == "test-project"
        assert data["service"] == "Compute Engine"
        assert data["category"] == "compute"
        assert data["cost_usd"] == 1250.50
        assert "timestamp" in data

    def test_cost_snapshot_top_services(self):
        """Top services are returned correctly."""
        snapshot = CostSnapshot(
            timestamp=datetime.now(UTC),
            total_cost_usd=3000.0,
            total_forecast_usd=3200.0,
            cost_by_category={},
            cost_by_service={
                "Compute Engine": 1500.0,
                "Cloud Storage": 800.0,
                "Cloud SQL": 500.0,
                "Cloud Load Balancing": 200.0,
            },
            cost_by_project={},
            cost_by_region={},
            anomalies_detected=[],
            data_freshness_minutes=0,
            collection_duration_seconds=1.0
        )

        top_5 = snapshot.get_top_services(5)

        assert len(top_5) <= 5
        assert top_5[0][0] == "Compute Engine"
        assert top_5[0][1] == 1500.0

    def test_cost_snapshot_to_dict(self):
        """Cost snapshot converts to dictionary correctly."""
        snapshot = CostSnapshot(
            timestamp=datetime.now(UTC),
            total_cost_usd=1000.0,
            total_forecast_usd=1050.0,
            cost_by_category={CostCategory.COMPUTE: 600.0},
            cost_by_service={"Compute Engine": 600.0},
            cost_by_project={"test-project": 1000.0},
            cost_by_region={"us-central1": 1000.0},
            anomalies_detected=[],
            data_freshness_minutes=0,
            collection_duration_seconds=1.5
        )

        data = snapshot.to_dict()

        assert data["total_cost_usd"] == 1000.0
        assert data["total_forecast_usd"] == 1050.0
        assert "timestamp" in data
        assert "top_services" in data


class TestCostManagementService:
    """Tests for Cost Management Service."""

    def test_service_initialization(self, cost_service):
        """Cost service initializes correctly."""
        assert cost_service.project_id == "test-ollama-prod"
        assert cost_service.monthly_budget_usd == 50000.0
        assert cost_service.current_month_cost == 0.0
        assert len(cost_service.alerts) == 0

    @pytest.mark.asyncio
    async def test_collect_and_update(self, cost_service):
        """collect_and_update returns valid snapshot."""
        snapshot = await cost_service.collect_and_update()

        assert isinstance(snapshot, CostSnapshot)
        assert snapshot.total_cost_usd > 0

    @pytest.mark.asyncio
    async def test_monthly_cost_tracking(self, cost_service):
        """Monthly cost is tracked correctly."""
        await cost_service.collect_and_update()

        assert cost_service.current_month_cost > 0

    def test_cost_summary_generation(self, cost_service):
        """Cost summary is generated correctly."""
        # Manually set some data
        cost_service.current_month_cost = 25000.0
        cost_service.collector.monthly_totals = {"2026-01": 25000.0}

        summary = cost_service.get_cost_summary()

        assert summary["current_month_cost_usd"] == 25000.0
        assert summary["monthly_budget_usd"] == 50000.0
        assert summary["budget_percent"] == 50.0
        assert summary["remaining_budget_usd"] == 25000.0
        assert summary["forecast_status"] == "on_track"

    def test_budget_alert_generation(self, cost_service):
        """Budget alerts are generated at threshold."""
        # Set cost to 85% of budget
        cost_service.current_month_cost = 42500.0

        cost_service._update_alerts(CostSnapshot(
            timestamp=datetime.now(UTC),
            total_cost_usd=42500.0,
            total_forecast_usd=45000.0,
            cost_by_category={},
            cost_by_service={},
            cost_by_project={},
            cost_by_region={},
            anomalies_detected=[],
            data_freshness_minutes=0,
            collection_duration_seconds=1.0
        ))

        assert len(cost_service.alerts) > 0
        budget_alert = next(
            (a for a in cost_service.alerts if a["type"] == "budget_alert"),
            None
        )
        assert budget_alert is not None
        assert budget_alert["severity"] == "warning"

    def test_cost_breakdown_generation(self, cost_service):
        """Cost breakdown is generated correctly."""
        # Add snapshot with data
        snapshot = CostSnapshot(
            timestamp=datetime.now(UTC),
            total_cost_usd=3000.0,
            total_forecast_usd=3200.0,
            cost_by_category={
                CostCategory.COMPUTE: 1500.0,
                CostCategory.STORAGE: 800.0,
            },
            cost_by_service={
                "Compute Engine": 1500.0,
                "Cloud Storage": 800.0,
            },
            cost_by_project={"test-project": 3000.0},
            cost_by_region={"us-central1": 3000.0},
            anomalies_detected=[],
            data_freshness_minutes=0,
            collection_duration_seconds=1.0
        )
        cost_service.collector.hourly_snapshots.append(snapshot)

        breakdown = cost_service.get_cost_breakdown()

        assert "by_category" in breakdown
        assert "by_service" in breakdown
        assert breakdown["by_service"]["Compute Engine"] == 1500.0

    def test_cost_trend_retrieval(self, cost_service):
        """Cost trend is retrieved correctly."""
        # Add snapshots
        now = datetime.now(UTC)
        for i in range(10):
            snapshot = CostSnapshot(
                timestamp=now - timedelta(hours=10-i),
                total_cost_usd=100.0 * (i + 1),
                total_forecast_usd=100.0 * (i + 1),
                cost_by_category={},
                cost_by_service={},
                cost_by_project={},
                cost_by_region={},
                anomalies_detected=[],
                data_freshness_minutes=0,
                collection_duration_seconds=1.0
            )
            cost_service.collector.hourly_snapshots.append(snapshot)

        trend = cost_service.get_cost_trend(hours=10)

        assert len(trend["data_points"]) > 0
        assert "min_cost_usd" in trend
        assert "max_cost_usd" in trend
        assert "avg_cost_usd" in trend

    def test_optimization_recommendations(self, cost_service):
        """Optimization recommendations are generated."""
        # Add snapshot with high compute costs
        snapshot = CostSnapshot(
            timestamp=datetime.now(UTC),
            total_cost_usd=6000.0,
            total_forecast_usd=6000.0,
            cost_by_category={
                CostCategory.COMPUTE: 4000.0,
                CostCategory.STORAGE: 1500.0,
                CostCategory.NETWORK: 300.0,
            },
            cost_by_service={},
            cost_by_project={},
            cost_by_region={},
            anomalies_detected=[],
            data_freshness_minutes=0,
            collection_duration_seconds=1.0
        )
        cost_service.collector.hourly_snapshots.append(snapshot)

        recommendations = cost_service.get_optimization_recommendations()

        # Should get recommendations for high compute
        assert len(recommendations) > 0
        compute_recs = [r for r in recommendations if r["category"] == "compute"]
        assert len(compute_recs) > 0

    @pytest.mark.asyncio
    async def test_comprehensive_cost_report(self, cost_service):
        """Comprehensive cost report is generated."""
        await cost_service.collect_and_update()

        report = await cost_service.generate_cost_report()

        assert "timestamp" in report
        assert "summary" in report
        assert "breakdown" in report
        assert "trend" in report
        assert "recommendations" in report
        assert "alerts" in report
