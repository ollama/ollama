# Issue #46: Predictive Cost Management Implementation Guide

**Issue**: [#46 - Predictive Cost Management](https://github.com/kushin77/ollama/issues/46)  
**Status**: OPEN - Ready for Assignment  
**Priority**: HIGH  
**Estimated Hours**: 80h (11.4 days)  
**Timeline**: Week 1-3 (Feb 3-21, 2026)  
**Dependencies**: #42 (Federation data collection)  
**Parallel Work**: #47, #48, #49, #50  

## Overview

Implement cost forecasting and optimization using **Prophet**, **GCP Cost Analysis**, and **anomaly detection**. Track infrastructure costs, predict future spending, and identify optimization opportunities.

## Architecture

```
GCP Billing Data → Cost Aggregator → Prophet Forecaster → Recommendations
     ↓
   Current Spending
     ↓
   Anomaly Detection
```

## Phase 1: Cost Data Collection (Week 1, 20 hours)

### 1.1 GCP Billing Integration
- Cloud Billing API integration
- BigQuery cost data export
- Hourly cost aggregation
- Resource-level cost attribution

**Code** (300 lines - `ollama/cost/gcp_cost_collector.py`):
```python
class GCPCostCollector:
    """Collects cost data from GCP Billing API."""

    async def collect_daily_costs(self) -> CostSnapshot:
        """Collect costs for past 24 hours."""
        # Query BigQuery for detailed costs
        costs_by_resource = await self._query_costs()
        
        # Aggregate by service, region, project
        aggregated = self._aggregate_costs(costs_by_resource)
        
        return CostSnapshot(
            timestamp=datetime.now(),
            total_cost=aggregated['total'],
            by_service=aggregated['by_service'],
            by_region=aggregated['by_region'],
            by_project=aggregated['by_project']
        )

    async def get_cost_history(self, days: int = 30) -> list[CostSnapshot]:
        """Get cost history for trend analysis."""
        query = f"""
        SELECT
            DATE(usage_start_time) as date,
            service.description,
            SUM(cost) as daily_cost
        FROM `{self.billing_table}`
        WHERE DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        GROUP BY date, service.description
        """
        return await self._execute_query(query)
```

### 1.2 Resource Tagging & Attribution
- Enforce GCP resource tags (team, application, cost-center)
- Cost allocation by tag
- Budget tracking
- Chargeback automation

### 1.3 Cost Baseline Establishment
- Current month spending
- YTD spending
- Monthly average
- Trend analysis

## Phase 2: Cost Forecasting (Week 2, 30 hours)

### 2.1 Prophet Time-Series Forecasting
- Historical cost data input
- Trend + seasonality detection
- 30-day, 90-day, 1-year forecasts
- Confidence intervals (80%, 95%)

**Code** (350 lines - `ollama/cost/prophet_forecaster.py`):
```python
from prophet import Prophet
import pandas as pd

class CostForecaster:
    """Uses Facebook Prophet for cost forecasting."""

    def forecast_costs(
        self,
        historical_costs: list[CostSnapshot],
        periods: int = 30,
        interval_width: float = 0.95
    ) -> CostForecast:
        """Forecast costs using Prophet."""
        # Prepare data for Prophet
        df = pd.DataFrame([
            {'ds': c.timestamp, 'y': c.total_cost}
            for c in historical_costs
        ])

        # Fit model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=interval_width,
            changepoint_prior_scale=0.05
        )
        model.fit(df)

        # Generate forecast
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        return CostForecast(
            forecast_date=datetime.now(),
            periods=periods,
            predictions=[
                {
                    'date': row['ds'],
                    'predicted_cost': row['yhat'],
                    'lower_bound': row['yhat_lower'],
                    'upper_bound': row['yhat_upper']
                }
                for _, row in forecast.tail(periods).iterrows()
            ]
        )
```

### 2.2 Anomaly Detection
- Deviation detection (>20% from baseline)
- Cost spike alerts
- Root cause analysis
- Automatic notifications

**Code** (200 lines - `ollama/cost/anomaly_detector.py`)

### 2.3 What-If Analysis
- Scaling impact simulation
- Service change impact
- Region change impact
- Recommendation engine

## Phase 3: Optimization & Recommendations (Week 3, 30 hours)

### 3.1 Cost Optimization Recommendations
- Idle resource detection
- Right-sizing recommendations
- Commitment discount analysis
- Reserved instance recommendations

**Recommendations Engine**:
- Analyze past 30 days
- Compare to baselines
- Generate 5-10 recommendations
- ROI calculation for each

### 3.2 Budget Alerts & Controls
- Monthly budget tracking
- Service-level budgets
- Alert thresholds (75%, 90%, 100%)
- Budget overage prevention

**Budget Policies**:
- Production: Hard limit at 110% of monthly budget
- Staging: Hard limit at budget
- Development: Warning only

### 3.3 Cost Dashboard & Reporting
- Real-time cost display
- Daily/weekly/monthly reports
- Forecasted vs actual comparison
- Recommendations dashboard

## Acceptance Criteria

- [ ] GCP Billing data collection automated
- [ ] Cost data stored and aggregated
- [ ] Prophet forecasts generating (30/90/365 day)
- [ ] Anomaly detection working (>20% threshold)
- [ ] Recommendations generating monthly
- [ ] Cost dashboard visible
- [ ] Budget alerts working
- [ ] <5% forecast error for 30-day horizon

## Testing Strategy

- Unit tests for collectors (20 tests)
- Forecast accuracy tests (15 tests)
- Anomaly detection tests (12 tests)
- Integration tests (10 tests)

## Success Metrics

- **Forecast Accuracy**: 90%+ for 30-day horizon
- **Anomaly Detection**: 95%+ precision
- **Cost Visibility**: 100% of resources tracked
- **Recommendation ROI**: >$50K/month identified
- **Budget adherence**: ±5% of forecasted

---

**Next Steps**: Assign to FinOps engineer, begin Week 1 (parallel with Federation)
