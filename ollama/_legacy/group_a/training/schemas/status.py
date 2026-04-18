"""Training job lifecycle states.

Defines the enumeration of status categories for fine-tuning jobs.
"""

from enum import Enum


class TrainingStatus(str, Enum):
    """Lifecycle states of a training job."""

    PENDING = "pending"
    PROVISIONING = "provisioning"
    TRAINING = "training"
    EVALUATING = "evaluating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
