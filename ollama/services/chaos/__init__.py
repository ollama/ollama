"""
Chaos Engineering Module
=========================

Provides chaos testing infrastructure for validating system resilience.

Supported Chaos Experiments:
    - Network delays (latency injection)
    - Service failures (pod crashes, connection drops)
    - Resource exhaustion (CPU, memory throttling)
    - Cascading failures (upstream failure propagation)
    - State inconsistency (database corruption simulation)

Usage:
    >>> from ollama.services.chaos import ChaosExperiment, ExperimentType
    >>> exp = ChaosExperiment(
    ...     name="inference_latency_spike",
    ...     experiment_type=ExperimentType.NETWORK_LATENCY,
    ...     duration_seconds=300,
    ...     target_service="inference"
    ... )
    >>> exp.run()

Example:
    >>> # Run chaos experiment during canary deployment
    >>> chaos_manager = ChaosManager()
    >>> chaos_manager.schedule_experiment(
    ...     experiment=exp,
    ...     trigger=ChaosSchedule.DURING_CANARY,
    ...     rollback_on_failure=True
    ... )
"""

from ollama.services.chaos.config import (
    ChaosConfig,
    ChaosExperiment,
    ComputeConfig,
    ExperimentType,
    FailureMode,
    NetworkConfig,
    SeverityLevel,
)
from ollama.services.chaos.executor import ChaosExecutor
from ollama.services.chaos.manager import ChaosManager, ExperimentState
from ollama.services.chaos.metrics import ChaosMetrics

__all__ = [
    "ChaosConfig",
    "ChaosExecutor",
    "ChaosExperiment",
    "ChaosManager",
    "ChaosMetrics",
    "ComputeConfig",
    "ExperimentState",
    "ExperimentType",
    "FailureMode",
    "NetworkConfig",
    "SeverityLevel",
]
