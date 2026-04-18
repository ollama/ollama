"""Schemas: EmbeddingsRequest for /embeddings route."""

from pydantic import BaseModel, Field


class EmbeddingsRequest(BaseModel):
    """Embeddings request model"""

    model: str = Field(
        default="all-minilm-l6-v2",
        description="Model name (all-minilm-l6-v2, all-mpnet-base-v2, etc)",
    )
    prompt: str = Field(..., description="Text to embed", min_length=1, max_length=1024)
