"""Training job creation schema.

Defines the request model for initiating a new fine-tuning job.
"""

from ollama.training.schemas.config import TrainingConfig
from pydantic import BaseModel, Field


class TrainingJobCreate(BaseModel):
    """Request schema to create a new training job."""

    base_model: str = Field(..., description="Name of the base model to fine-tune")
    dataset_id: str = Field(..., description="ID of the pre-uploaded dataset")
    config: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training hyperparameters"
    )
    description: str | None = Field(None, description="Optional job description")
