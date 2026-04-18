"""Schemas: SemanticSearchRequest for /semantic-search route."""

from pydantic import BaseModel, Field


class SemanticSearchRequest(BaseModel):
    """Semantic search request"""

    collection: str = Field(..., description="Qdrant collection name")
    query: str = Field(..., description="Query text")
    model: str = Field(default="all-minilm-l6-v2", description="Embedding model")
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum similarity score"
    )
