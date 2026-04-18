"""Schemas: SemanticSearchResponse for /semantic-search route."""

from pydantic import BaseModel

from ollama.api.schemas.search_result import SearchResult


class SemanticSearchResponse(BaseModel):
    """Semantic search response"""

    query: str
    results: list[SearchResult]
    count: int
