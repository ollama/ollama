"""Text generation request model."""

from dataclasses import dataclass


@dataclass
class GenerateRequest:
    """Request for text generation."""

    model: str
    """Model name to use for generation"""

    prompt: str
    """Input prompt for generation"""

    system: str | None = None
    """System prompt to set context"""

    temperature: float = 0.7
    """Sampling temperature (0.0 to 2.0)"""

    top_p: float = 0.9
    """Nucleus sampling parameter"""

    top_k: int = 40
    """Top-K sampling parameter"""

    repeat_penalty: float = 1.1
    """Penalty for repetition"""

    num_predict: int = 100
    """Maximum tokens to generate"""

    stop: list[str] | None = None
    """Optional list of stop sequences"""

    context_length: int = 2048
    """Context window size"""

    stream: bool = False
    """Whether to stream response"""
