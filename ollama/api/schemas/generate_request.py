"""Schemas: GenerateRequest for /generate route."""

from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Generate request model"""

    model: str = Field(..., description="Model name to use for generation")
    prompt: str = Field(..., description="Prompt text")
    stream: bool = Field(default=False, description="Stream response")
    options: dict[str, Any] | None = Field(default=None, description="Generation options")
