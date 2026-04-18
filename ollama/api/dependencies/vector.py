"""Vector manager dependency helpers."""

import logging

from fastapi import HTTPException

from ollama.services.models.resilient_vector import ResilientVectorManager
from ollama.services.models.vector import VectorManager

log = logging.getLogger(__name__)

# Global vector manager instance
_vector_manager: VectorManager | ResilientVectorManager | None = None


async def get_vector_manager() -> VectorManager | ResilientVectorManager:
    """FastAPI dependency that yields the vector manager."""
    global _vector_manager
    if _vector_manager is None:
        log.warning("Vector manager not initialized - returning 503")
        raise HTTPException(
            status_code=503,
            detail="Vector manager not initialized. Service is starting up.",
        )
    return _vector_manager


def set_global_vector_manager(manager: VectorManager | ResilientVectorManager) -> None:
    """Set the global vector manager instance (called from main.py startup)."""
    global _vector_manager
    _vector_manager = manager
    log.info("Global vector manager dependency initialized")
