"""Semantic Cache Service - Qdrant + Redis powered fuzzy matching.

Provides ability to retrieve cached responses for semantically similar prompts,
extending the exact-match Redis cache.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import structlog
from qdrant_client.http.models import PointStruct

from ollama.services.cache.cache import CacheManager
from ollama.services.inference.ollama_client_main import OllamaClient
from ollama.services.models.vector import VectorManager

log = structlog.get_logger(__name__)


class SemanticCache:
    """Semantic Cache using Vector Similarity + Redis Storage."""

    COLLECTION_NAME = "semantic_cache"
    VECTOR_SIZE = 4096  # Default for Llama 3 / Mistral, adjust if using different embedding model
    THRESHOLD = 0.95

    def __init__(
        self,
        cache_manager: CacheManager,
        vector_manager: VectorManager,
        ollama_client: OllamaClient,
        embedding_model: str = "llama3.2",
    ) -> None:
        """Initialize semantic cache.

        Args:
            cache_manager: Exact-match Redis cache manager.
            vector_manager: Qdrant vector manager.
            ollama_client: Client for generating embeddings.
            embedding_model: Model to use for embeddings.
        """
        self.cache = cache_manager
        self.vector = vector_manager
        self.ollama = ollama_client
        self.embedding_model = embedding_model
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize semantic cache collection."""
        if self._initialized:
            return

        # Ensure collection exists
        await self.vector.create_collection(
            collection_name=self.COLLECTION_NAME,
            vector_size=self.VECTOR_SIZE,
        )
        self._initialized = True
        log.info("semantic_cache_initialized", collection=self.COLLECTION_NAME)

    def _get_prompt_hash(self, prompt: str) -> str:
        """Generate a unique hash for a prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    async def get(self, prompt: str) -> Any | None:
        """Get semantically similar response from cache.

        Args:
            prompt: User prompt.

        Returns:
            Cached response or None if No match found.
        """
        try:
            # 1. Generate embedding
            embedding = await self.ollama.generate_embeddings(self.embedding_model, prompt)

            # 2. Search Qdrant
            results = await self.vector.search_vectors(
                collection_name=self.COLLECTION_NAME,
                query_vector=embedding,
                limit=1,
                score_threshold=self.THRESHOLD,
            )

            if not results:
                return None

            # 3. Get Redis key from payload
            match = results[0]
            redis_key = match.payload.get("redis_key")
            if not redis_key:
                return None

            log.info("semantic_cache_hit", score=match.score, prompt=prompt[:50])

            # 4. Retrieve from Redis
            return await self.cache.get(redis_key)

        except Exception as e:
            log.error("semantic_cache_get_failed", error=str(e))
            return None

    async def set(self, prompt: str, response: Any, ttl: int = 3600) -> bool:
        """Store response in semantic cache.

        Args:
            prompt: User prompt.
            response: Response to cache.
            ttl: Time to live in seconds.

        Returns:
            True if successful.
        """
        try:
            # 1. Generate embedding and hash
            embedding = await self.ollama.generate_embeddings(self.embedding_model, prompt)
            prompt_hash = self._get_prompt_hash(prompt)
            redis_key = f"semantic:{prompt_hash}"

            # 2. Store response in Redis
            await self.cache.set(redis_key, response, ttl=ttl)

            # 3. Upsert to Qdrant
            point = PointStruct(
                id=prompt_hash[:32],  # Use first 32 chars of hash as UUID-like ID
                vector=embedding,
                payload={
                    "prompt": prompt[:100],  # Store snippet for debugging
                    "redis_key": redis_key,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            await self.vector.upsert_vectors(self.COLLECTION_NAME, [point])

            log.info("semantic_cache_stored", key=redis_key)
            return True

        except Exception as e:
            log.error("semantic_cache_set_failed", error=str(e))
            return False
