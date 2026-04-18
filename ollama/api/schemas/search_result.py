"""Schemas: SearchResult for semantic search response."""

from pydantic import BaseModel


class SearchResult(BaseModel):
    """Search result item"""

    id: str
    score: float
    text: str | None = None
