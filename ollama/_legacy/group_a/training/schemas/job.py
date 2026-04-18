"""Full training job representation.

Defines the complete schema for a training job including metrics and metadata.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from ollama.training.schemas.config import TrainingConfig
from ollama.training.schemas.status import TrainingStatus
from pydantic import BaseModel, ConfigDict, Field


class TrainingJob(BaseModel):
    """Full representation of a training job."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "base_model": "llama3",
                "dataset_id": "ds_customer_service_v1",
                "status": "training",
                "progress": 45.5,
                "config": {"learning_rate": 2e-5, "batch_size": 2, "quantization": "4bit"},
            }
        },
    )

    id: UUID = Field(default_factory=uuid4)
    base_model: str
    dataset_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
