"""Schemas: Chat message representation."""

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message"""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
