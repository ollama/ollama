"""Schemas: EmbeddingsResponse for /embeddings route."""

from pydantic import BaseModel, Field


class EmbeddingsResponse(BaseModel):
    """Embeddings response model"""

    embedding: list[float] = Field(..., description="Vector embedding")
    model: str = Field(..., description="Model used")
    dimensions: int = Field(..., description="Embedding dimensions")
