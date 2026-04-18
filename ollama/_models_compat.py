"""Ollama Database Models (backward compatibility re-exports).

This module re-exports ORM models split into individual modules. New code should
import models directly from their dedicated modules under `ollama.models`.

Legacy imports (deprecated):
    >>> from ollama.models import Base, User, APIKey

Preferred imports:
    >>> from ollama.models import Base
    >>> from ollama.models import User
    >>> from ollama.models import APIKey
    >>> from ollama.models import Conversation
    >>> from ollama.models import Message
    >>> from ollama.models import Document
    >>> from ollama.models import Usage
"""

from ollama.models import APIKey, Base, Conversation, Document, Message, Usage, User

__all__ = ["APIKey", "Base", "Conversation", "Document", "Message", "Usage", "User"]
