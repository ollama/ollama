"""Resource management types.

Defines the categories of workloads that compete for hardware resources.
"""

from enum import Enum


class WorkloadType(str, Enum):
    """Types of workloads competing for resources."""

    INFERENCE = "inference"
    TRAINING = "training"
