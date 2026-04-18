"""Chat message representation."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ChatMessage:
    """Single message in chat conversation.

    Represents a message from either user or assistant in a conversation.
    """

    role: Literal["user", "assistant", "system"]
    """Message author role."""

    content: str
    """Message content/text."""
