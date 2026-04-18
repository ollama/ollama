"""Persistence service module.

Handles data persistence including database operations, ORM models, and request/response
serialization. Manages conversations, chat history, and document storage.

This module provides the data access layer for the application.
"""

from ollama.services.persistence.chat_message import ChatMessage
from ollama.services.persistence.chat_request import ChatRequest
from ollama.services.persistence.database import (
    DatabaseManager,
    get_db,
    get_db_manager,
    init_database,
)

Database = DatabaseManager

__all__: list[str] = [
    "ChatMessage",
    "ChatRequest",
    "Database",
    "DatabaseManager",
    "get_cache_manager",
    "get_db",
    "get_db_manager",
    "init_cache",
    "init_database",
]
