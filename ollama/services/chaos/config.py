"""
Chaos Engineering Configuration Models
=======================================

Defines all configuration models for chaos experiments, failure modes,
and resilience validation.

Provides:
    - Experiment type classification
    - Failure mode definitions
    - Experiment configuration
    - Network chaos parameters
    - Compute resource chaos parameters
    - Validation and constraints

Example:
    >>> from ollama.services.chaos.config import (
    ...     ChaosExperiment, ExperimentType, NetworkConfig
    ... )
    >>> exp = ChaosExperiment(
    ...     name="inference_latency",
    ...     experiment_type=ExperimentType.NETWORK_LATENCY,
    ...     target_service="inference",
    ...     duration_seconds=300,
    ...     network_config=NetworkConfig(
    ...         latency_ms=200,
    ...         jitter_ms=50
    ...     )
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, root_validator, validator


class ExperimentType(str, Enum):
    """Type of chaos experiment."""

    NETWORK_LATENCY = "network_latency"  # Inject latency
    NETWORK_LOSS = "network_loss"  # Drop packets
    SERVICE_FAILURE = "service_failure"  # Pod crash/restart
    RESOURCE_CPU = "resource_cpu"  # CPU throttling
    RESOURCE_MEMORY = "resource_memory"  # Memory exhaustion
    CASCADING_FAILURE = "cascading_failure"  # Upstream failure
    STATE_INCONSISTENCY = "state_inconsistency"  # Data corruption
    DEPENDENCY_LATENCY = "dependency_latency"  # Slow dependency


class SeverityLevel(str, Enum):
    """Severity level of experiment impact."""

    LOW = "low"  # <5% request failure
    MEDIUM = "medium"  # 5-20% failure
    HIGH = "high"  # 20-50% failure
    CRITICAL = "critical"  # >50% failure


class FailureMode(str, Enum):
    """Expected failure mode during experiment."""

    GRACEFUL_DEGRADATION = "graceful_degradation"  # Service continues, reduced capacity
    CIRCUIT_BREAKER_TRIP = "circuit_breaker_trip"  # Circuit breaker opens
    TIMEOUT = "timeout"  # Requests timeout
    RETRY_EXHAUSTION = "retry_exhaustion"  # All retries fail
    FALLBACK_ACTIVATED = "fallback_activated"  # Fallback mechanism activates
    QUEUE_BUILDUP = "queue_buildup"  # Requests queue
    CASCADING_FAILURE = "cascading_failure"  # Failure propagates upstream


class NetworkConfig(BaseModel):
    """Network chaos configuration."""

    latency_ms: int = Field(
        default=0,
        description="Network latency to inject (milliseconds)",
        ge=0,
        le=5000,
    )

    jitter_ms: int = Field(
        default=0,
        description="Latency jitter (random variance)",
        ge=0,
        le=1000,
    )

    packet_loss_percent: float = Field(
        default=0.0,
        description="Packet loss percentage",
        ge=0.0,
        le=100.0,
    )

    bandwidth_limit_mbps: int | None = Field(
        default=None,
        description="Bandwidth limit (Mbps)",
        ge=1,
    )

    target_ports: list[int] = Field(
        default_factory=lambda: [5432, 6379, 8000],
        description="Target ports for chaos",
    )

    @validator("latency_ms")
    def validate_latency(cls, v: int) -> int:
        """Validate latency is reasonable."""
        if v > 0 and v < 10:
            raise ValueError("Minimum latency should be 10ms for measurable impact")
        return v


class ComputeConfig(BaseModel):
    """Compute resource chaos configuration."""

    cpu_percent: float | None = Field(
        default=None,
        description="CPU throttle percentage (0-100)",
        ge=0,
        le=100,
    )

    memory_mb: int | None = Field(
        default=None,
        description="Memory limit (MB)",
        ge=128,
    )

    io_limit_mbps: int | None = Field(
        default=None,
        description="I/O throughput limit (Mbps)",
        ge=1,
    )

    target_containers: list[str] = Field(
        default_factory=lambda: ["api", "inference"],
        description="Target containers for chaos",
    )

    graceful_period_seconds: int = Field(
        default=30,
        description="Grace period before killing process",
        ge=5,
    )


class HealthCheck(BaseModel):
    """Health check configuration during chaos."""

    enabled: bool = Field(default=True, description="Enable health checks")

    interval_seconds: int = Field(
        default=5,
        description="Health check interval",
        ge=1,
    )

    timeout_seconds: int = Field(
        default=10,
        description="Health check timeout",
        ge=1,
    )

    unhealthy_threshold: int = Field(
        default=3,
        description="Consecutive failures before marking unhealthy",
        ge=1,
    )

    abort_on_unhealthy: bool = Field(
        default=True,
        description="Abort experiment if service becomes unhealthy",
    )


class RollbackConfig(BaseModel):
    """Rollback configuration."""

    enabled: bool = Field(default=True, description="Enable automatic rollback")

    on_failure: bool = Field(
        default=True,
        description="Rollback on experiment failure",
    )

    on_threshold_breach: bool = Field(
        default=True,
        description="Rollback if error rate exceeds threshold",
    )

    error_rate_threshold: float = Field(
        default=0.10,
        description="Error rate threshold for rollback",
        ge=0.0,
        le=1.0,
    )

    max_duration_seconds: int = Field(
        default=600,
        description="Maximum experiment duration before auto-rollback",
        ge=60,
    )


class ChaosExperiment(BaseModel):
    """Complete chaos experiment definition."""

    name: str = Field(description="Experiment name")

    description: str | None = Field(
        default=None,
        description="Detailed description",
    )

    experiment_type: ExperimentType = Field(description="Type of chaos experiment")

    target_service: str = Field(description="Service to target (api, inference, etc)")

    severity: SeverityLevel = Field(
        default=SeverityLevel.MEDIUM,
        description="Expected severity",
    )

    duration_seconds: int = Field(
        default=300,
        description="Experiment duration",
        ge=60,
        le=3600,
    )

    network_config: NetworkConfig | None = Field(
        default=None,
        description="Network chaos config",
    )

    compute_config: ComputeConfig | None = Field(
        default=None,
        description="Compute resource chaos config",
    )

    health_check: HealthCheck = Field(
        default_factory=HealthCheck,
        description="Health check config",
    )

    rollback_config: RollbackConfig = Field(
        default_factory=RollbackConfig,
        description="Rollback config",
    )

    expected_failure_modes: list[FailureMode] = Field(
        default_factory=list,
        description="Expected failure modes",
    )

    owner: str = Field(description="Experiment owner (team name)")

    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Metadata tags",
    )

    created_at: datetime = Field(default_factory=datetime.now)

    @root_validator(skip_on_failure=True)
    def validate_experiment_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate experiment has correct configuration for type."""
        exp_type = values.get("experiment_type")

        if exp_type in [ExperimentType.NETWORK_LATENCY, ExperimentType.NETWORK_LOSS]:
            if not values.get("network_config"):
                raise ValueError(f"{exp_type} requires network_config")

        if exp_type in [
            ExperimentType.RESOURCE_CPU,
            ExperimentType.RESOURCE_MEMORY,
        ]:
            if not values.get("compute_config"):
                raise ValueError(f"{exp_type} requires compute_config")

        return values


class ChaosConfig(BaseModel):
    """Global chaos engineering configuration."""

    experiments: dict[str, ChaosExperiment] = Field(
        default_factory=dict,
        description="Registered experiments",
    )

    enabled: bool = Field(
        default=True,
        description="Enable chaos engineering",
    )

    run_in_production: bool = Field(
        default=False,
        description="Allow chaos in production (requires approval)",
    )

    canary_trigger_enabled: bool = Field(
        default=True,
        description="Auto-trigger during canary deployments",
    )

    metrics_collection_enabled: bool = Field(
        default=True,
        description="Collect metrics during experiments",
    )

    max_concurrent_experiments: int = Field(
        default=1,
        description="Max concurrent experiments",
        ge=1,
        le=5,
    )

    def add_experiment(self, experiment: ChaosExperiment) -> None:
        """Add chaos experiment.

        Args:
            experiment: Experiment to add

        Raises:
            ValueError: If experiment name already exists
        """
        if experiment.name in self.experiments:
            raise ValueError(f"Experiment '{experiment.name}' already exists")

        self.experiments[experiment.name] = experiment

    def remove_experiment(self, name: str) -> ChaosExperiment | None:
        """Remove chaos experiment.

        Args:
            name: Experiment name

        Returns:
            Removed experiment or None
        """
        return self.experiments.pop(name, None)

    def get_experiment(self, name: str) -> ChaosExperiment | None:
        """Get experiment by name.

        Args:
            name: Experiment name

        Returns:
            Experiment or None
        """
        return self.experiments.get(name)

    def list_experiments(
        self,
        experiment_type: ExperimentType | None = None,
        target_service: str | None = None,
    ) -> list[ChaosExperiment]:
        """List experiments with optional filtering.

        Args:
            experiment_type: Filter by type
            target_service: Filter by target service

        Returns:
            List of experiments
        """
        experiments = list(self.experiments.values())

        if experiment_type:
            experiments = [e for e in experiments if e.experiment_type == experiment_type]

        if target_service:
            experiments = [e for e in experiments if e.target_service == target_service]

        return experiments

    def validate_for_environment(self, environment: str) -> bool:
        """Validate configuration for environment.

        Args:
            environment: Environment name (dev, staging, prod)

        Returns:
            True if valid for environment

        Raises:
            ValueError: If configuration invalid for environment
        """
        if environment == "production" and not self.run_in_production:
            raise ValueError("Chaos engineering disabled for production")

        return True


def get_default_chaos_config() -> ChaosConfig:
    """Get default chaos configuration with sample experiments.

    Returns:
        Default configuration with 5 sample experiments

    Example:
        >>> config = get_default_chaos_config()
        >>> print(f"Configured {len(config.experiments)} experiments")
    """
    config = ChaosConfig(
        enabled=True,
        run_in_production=False,
        canary_trigger_enabled=True,
    )

    # Network latency experiment
    config.add_experiment(
        ChaosExperiment(
            name="inference_latency_spike",
            description="Inject 200ms latency to inference service",
            experiment_type=ExperimentType.NETWORK_LATENCY,
            target_service="inference",
            severity=SeverityLevel.MEDIUM,
            duration_seconds=300,
            network_config=NetworkConfig(
                latency_ms=200,
                jitter_ms=50,
            ),
            expected_failure_modes=[
                FailureMode.GRACEFUL_DEGRADATION,
                FailureMode.TIMEOUT,
            ],
            owner="platform-team",
        )
    )

    # Packet loss experiment
    config.add_experiment(
        ChaosExperiment(
            name="network_loss_simulation",
            description="Simulate 5% packet loss to database",
            experiment_type=ExperimentType.NETWORK_LOSS,
            target_service="database",
            severity=SeverityLevel.MEDIUM,
            duration_seconds=300,
            network_config=NetworkConfig(
                packet_loss_percent=5.0,
                target_ports=[5432],
            ),
            expected_failure_modes=[
                FailureMode.RETRY_EXHAUSTION,
                FailureMode.CIRCUIT_BREAKER_TRIP,
            ],
            owner="database-team",
        )
    )

    # CPU throttling experiment
    config.add_experiment(
        ChaosExperiment(
            name="cpu_exhaustion_test",
            description="Throttle API CPU to 50%",
            experiment_type=ExperimentType.RESOURCE_CPU,
            target_service="api",
            severity=SeverityLevel.HIGH,
            duration_seconds=300,
            compute_config=ComputeConfig(
                cpu_percent=50.0,
                target_containers=["api"],
            ),
            expected_failure_modes=[
                FailureMode.GRACEFUL_DEGRADATION,
                FailureMode.QUEUE_BUILDUP,
            ],
            owner="platform-team",
        )
    )

    # Memory exhaustion experiment
    config.add_experiment(
        ChaosExperiment(
            name="memory_pressure_test",
            description="Limit inference container to 512MB",
            experiment_type=ExperimentType.RESOURCE_MEMORY,
            target_service="inference",
            severity=SeverityLevel.HIGH,
            duration_seconds=300,
            compute_config=ComputeConfig(
                memory_mb=512,
                target_containers=["inference"],
            ),
            expected_failure_modes=[
                FailureMode.TIMEOUT,
                FailureMode.CASCADING_FAILURE,
            ],
            owner="ml-platform-team",
        )
    )

    # Service failure experiment
    config.add_experiment(
        ChaosExperiment(
            name="cache_failure_recovery",
            description="Kill Redis pod and observe recovery",
            experiment_type=ExperimentType.SERVICE_FAILURE,
            target_service="cache",
            severity=SeverityLevel.HIGH,
            duration_seconds=300,
            compute_config=ComputeConfig(
                target_containers=["redis"],
                graceful_period_seconds=30,
            ),
            expected_failure_modes=[
                FailureMode.FALLBACK_ACTIVATED,
                FailureMode.CIRCUIT_BREAKER_TRIP,
            ],
            owner="platform-team",
        )
    )

    return config


# Global chaos configuration instance
_chaos_config: ChaosConfig | None = None


def get_chaos_config() -> ChaosConfig:
    """Get or initialize global chaos configuration.

    Returns:
        Global chaos configuration

    Example:
        >>> config = get_chaos_config()
        >>> experiments = config.list_experiments()
    """
    global _chaos_config
    if _chaos_config is None:
        _chaos_config = get_default_chaos_config()
    return _chaos_config


def set_chaos_config(config: ChaosConfig) -> None:
    """Set global chaos configuration.

    Args:
        config: Chaos configuration to set

    Example:
        >>> custom_config = ChaosConfig(...)
        >>> set_chaos_config(custom_config)
    """
    global _chaos_config
    _chaos_config = config
