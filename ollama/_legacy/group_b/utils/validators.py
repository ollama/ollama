"""Validation utilities for Ollama API."""

import re


def validate_prompt(prompt: str, max_length: int = 5000) -> bool:
    """Validate prompt text.

    Args:
        prompt: Text to validate.
        max_length: Maximum allowed length.

    Returns:
        True if valid, False otherwise.
    """
    if not prompt or not prompt.strip():
        return False
    if len(prompt) > max_length:
        return False
    return True


def validate_model_name(name: str, max_length: int = 100) -> bool:
    """Validate model name format.

    Args:
        name: Model name to validate.
        max_length: Maximum allowed length.

    Returns:
        True if valid, False otherwise.
    """
    if not name or not name.strip():
        return False
    if len(name) > max_length:
        return False
    # Pattern: {environment}-{application}-{component} or simple names
    # Broadened to match existing models like llama3.2
    pattern = r"^[a-zA-Z0-9.\-_:]+$"
    return bool(re.match(pattern, name))


def validate_temperature(value: float) -> bool:
    """Validate temperature parameter.

    Args:
        value: Temperature value (typically 0.0 to 2.0).

    Returns:
        True if valid, False otherwise.
    """
    return 0.0 <= value <= 2.0


def validate_max_tokens(tokens: int) -> None:
    """Validate that max_tokens is a positive integer.

    Args:
        tokens: Number of tokens to validate.

    Raises:
        ValueError: If tokens is invalid.
    """
    if not isinstance(tokens, int):
        raise ValueError("max_tokens must be an integer")
    if tokens <= 0:
        raise ValueError("max_tokens must be positive")
