"""API route handlers.

Defines HTTP route handlers for all API endpoints. Routes are organized by resource
and delegate to service layer for business logic. Each route module handles a specific
resource with CRUD operations and domain-specific endpoints.

This module provides the HTTP interface to the application.
"""

from ollama.api.routes import (
    chat,
    conversations,
    documents,
    embeddings,
    generate,
    health,
    models,
    usage,
)

__all__ = [
    "health",
    "models",
    "generate",
    "chat",
    "embeddings",
    "conversations",
    "documents",
    "usage",
]
