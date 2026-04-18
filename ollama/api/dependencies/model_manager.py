"""Model manager dependency helpers."""

import logging

from fastapi import HTTPException

from ollama.services.inference.ollama_client_main import OllamaClient
from ollama.services.models.ollama_model_manager import OllamaModelManager

log = logging.getLogger(__name__)


# Global model manager instance
_model_manager: OllamaModelManager | None = None


async def close_model_manager() -> None:
    """Close the model manager on shutdown."""

    global _model_manager
    if _model_manager is not None:
        await _model_manager.close()
        log.info("Model manager closed")


async def get_model_manager() -> OllamaModelManager:
    """FastAPI dependency that yields the model manager."""

    if _model_manager is None:
        log.warning("Model manager not initialized - returning 503")
        raise HTTPException(
            status_code=503,
            detail="Model manager not initialized. Service is starting up.",
        )
    return _model_manager


def initialize_model_manager(base_url: str = "http://ollama:11434") -> None:
    """Initialize the global model manager instance."""

    global _model_manager
    client = OllamaClient(base_url=base_url)
    _model_manager = OllamaModelManager(ollama_client=client)
    log.info("Model manager initialized with base URL: %s", base_url)
