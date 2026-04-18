"""Inference API Schemas.

Consolidated schemas for AI model inference operations.
"""

from typing import Any

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """Schema for a single message in a conversation."""

    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


# Backwards-compatible aliases expected by tests and other modules
# Historically the project used ChatMessage/ChatRequest names.
# Provide aliases to avoid widespread refactors.
ChatMessage = ConversationMessage


class ConversationRequest(BaseModel):
    """Schema for continuing a conversation."""

    messages: list[ConversationMessage]
    model: str = Field(..., description="Model name")
    stream: bool = Field(True, description="Stream response")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1)
    num_predict: int = Field(128, ge=1)
    stop: list[str] | None = Field(None, description="Stop sequences")
    context: list[int] | None = Field(None, description="Context tokens")


# Backwards-compatible alias
ChatRequest = ConversationRequest


class ChatResponse(BaseModel):
    """Schema for chat completion response."""

    model: str
    message: dict[str, Any]
    done: bool
    total_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


class EmbeddingRequest(BaseModel):
    """Schema for embedding generation request."""

    model: str = Field(..., description="Model name")
    text: str = Field(..., description="Input text to embed")


class EmbeddingResponse(BaseModel):
    """Schema for embedding generation response."""

    model: str
    embedding: list[float]
    dimensions: int


class GenerateRequest(BaseModel):
    """Schema for text generation request."""

    model: str = Field(..., description="Model name")
    prompt: str = Field(..., description="Input prompt")
    system: str | None = Field(None, description="System prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1)
    repeat_penalty: float = Field(1.1, ge=0.0)
    num_predict: int = Field(100, ge=1, le=4096)
    stream: bool = Field(True)


class GenerateResponse(BaseModel):
    """Schema for text generation response."""

    model: str
    created_at: str | None = None
    prompt: str | None = None
    response: str
    done: bool
    context: list[int] | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class ListModelsResponse(BaseModel):
    """Schema for listing available models."""

    models: list[dict[str, Any]]
    total: int


class ModelPullRequest(BaseModel):
    """Schema for pulling a new model."""

    model_name: str = Field(..., description="Name of the model to pull")
    insecure: bool = Field(False, description="Allow insecure connections")
    stream: bool = Field(True, description="Stream progress updates")
