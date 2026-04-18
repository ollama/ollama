"""Chat completion request."""

from dataclasses import dataclass, field

from ollama.services.persistence.chat_message import ChatMessage


@dataclass
class ChatRequest:
    """Request for chat completion.

    Contains messages and model parameters for chat-based inference.
    """

    model: str
    """Model name for chat completion."""

    messages: list[ChatMessage]
    """Conversation messages."""

    temperature: float = 0.7
    """Sampling temperature (0.0-2.0)."""

    top_p: float = 0.9
    """Nucleus sampling parameter."""

    top_k: int = 40
    """Top-k sampling parameter."""

    num_predict: int = 128
    """Maximum tokens to generate."""

    stop: list[str] = field(default_factory=list)
    """Stop sequences."""

    stream: bool = False
    """Whether to stream generated tokens."""

    context: list[int] = field(default_factory=list)
    """Prompt context tokens."""
