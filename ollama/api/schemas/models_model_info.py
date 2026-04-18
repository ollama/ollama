"""Schemas: ModelInfo for /models routes."""

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Model information"""

    name: str
    size: str
    digest: str
    modified_at: str
