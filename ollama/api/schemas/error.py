"""Error response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error context")


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = Field(False, description="Whether the request was successful")
    error: ErrorDetail = Field(..., description="Error details")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Response metadata")
