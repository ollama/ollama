"""Schemas: ChatResponse for /chat route."""

from pydantic import BaseModel

from ollama.api.schemas.chat_message import Message


class ChatResponse(BaseModel):
    """Chat response model"""

    model: str
    created_at: str
    message: Message
    done: bool
