"""Schemas: GenerateResponse for /generate route."""

from pydantic import BaseModel


class GenerateResponse(BaseModel):
    """Generate response model"""

    model: str
    created_at: str
    response: str
    done: bool
