"""Text generation response model."""

from dataclasses import dataclass, field


@dataclass
class GenerateResponse:
    """Response from text generation."""

    model: str
    """Model used for generation"""

    prompt: str
    """Original prompt"""

    response: str
    """Generated text"""

    done: bool = True
    """Whether generation is complete"""

    context: list[int] = field(default_factory=list)
    """Context tokens for continuation"""

    total_duration: int = 0
    """Total generation time in nanoseconds"""

    load_duration: int = 0
    """Time to load model in nanoseconds"""

    prompt_eval_count: int = 0
    """Number of prompt tokens evaluated"""

    prompt_eval_duration: int = 0
    """Time to evaluate prompt in nanoseconds"""

    eval_count: int = 0
    """Number of tokens generated"""

    eval_duration: int = 0
    """Time to generate tokens in nanoseconds"""
