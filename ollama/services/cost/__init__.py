"""
Cost Management Module - Issue #46

Comprehensive GCP cost tracking and optimization.

Provides:
- Real-time cost collection from GCP Billing API
- Cost forecasting with Prophet
- Budget tracking and alerts
- Cost optimization recommendations
- Multi-project support
- Anomaly detection

Version: 1.0.0 (Week 1)
Status: PRODUCTION-READY
"""

from ollama.services.cost.collector import (
    CostCategory,
    CostSample,
    CostSnapshot,
    CostTimeGranularity,
    GCPCostCollector,
    ResourceMetric,
)
from ollama.services.cost.service import CostManagementService

__version__ = "1.0.0"
__status__ = "PRODUCTION-READY"

__all__ = [
    "GCPCostCollector",
    "CostManagementService",
    "CostSample",
    "CostSnapshot",
    "CostCategory",
    "CostTimeGranularity",
    "ResourceMetric",
]
