"""Schemas: ListModelsResponse for /models routes."""

from pydantic import BaseModel

from ollama.api.schemas.models_model_info import ModelInfo


class ListModelsResponse(BaseModel):
    """Response for list models"""

    models: list[ModelInfo]
