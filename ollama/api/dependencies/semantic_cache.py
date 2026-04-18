"""Semantic cache dependency helpers."""

import logging
from typing import Annotated

from fastapi import Depends

from ollama.api.dependencies.cache import get_cache_manager
from ollama.api.dependencies.model_manager import get_model_manager
from ollama.api.dependencies.vector import get_vector_manager
from ollama.services.cache.cache import CacheManager
from ollama.services.cache.semantic_cache import SemanticCache
from ollama.services.models.ollama_model_manager import OllamaModelManager
from ollama.services.models.vector import VectorManager

log = logging.getLogger(__name__)

# Global semantic cache instance
_semantic_cache: SemanticCache | None = None


async def get_semantic_cache(
    cache_manager: Annotated[CacheManager, Depends(get_cache_manager)],
    vector_manager: Annotated[VectorManager, Depends(get_vector_manager)],
    model_manager: Annotated[OllamaModelManager, Depends(get_model_manager)],
) -> SemanticCache:
    """FastAPI dependency that yields the semantic cache service."""
    global _semantic_cache

    if _semantic_cache is None:
        # Note: In production, embedding_model should come from settings
        _semantic_cache = SemanticCache(
            cache_manager=cache_manager,
            vector_manager=vector_manager,
            ollama_client=model_manager.client,
            embedding_model="llama3.2",
        )
        await _semantic_cache.initialize()
        log.info("Semantic cache dependency initialized")

    return _semantic_cache
