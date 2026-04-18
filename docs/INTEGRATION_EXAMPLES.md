"""Integration examples for Circuit Breaker and Response Caching.

Shows how to integrate the new resilience patterns and caching into
the Ollama inference API endpoints and services.
"""

# Example 1: Circuit Breaker Integration with OllamaClient

# ========================================================

from ollama.services.inference.ollama_client_main import OllamaClient
from ollama.services.resilience import get_circuit_breaker_manager
from ollama.exceptions.circuit_breaker import CircuitBreakerError

class ResilientOllamaClient:
"""Ollama client wrapper with circuit breaker protection."""

    def __init__(self, base_url: str = "http://ollama:11434") -> None:
        """Initialize resilient client.

        Args:
            base_url: Base URL for Ollama API.
        """
        self.client = OllamaClient(base_url=base_url)
        manager = get_circuit_breaker_manager()
        self.breaker = manager.get_or_create(
            service_name="ollama",
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=2,
        )

    async def generate(self, model: str, prompt: str) -> str:
        """Generate text with circuit breaker protection.

        Args:
            model: Model name.
            prompt: User prompt.

        Returns:
            Generated text.

        Raises:
            CircuitBreakerError: If service is unavailable.
        """
        try:
            response = self.breaker.call(
                self.client.generate_completion,
                model=model,
                prompt=prompt,
            )
            return response
        except CircuitBreakerError as e:
            # Service is known to be down, fail fast
            raise
        except Exception as e:
            # Other errors, log and raise
            raise

# Example 2: Response Cache Integration with API Endpoint

# ========================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ollama.services.cache.cache import CacheManager
from ollama.services.cache.response_cache import ResponseCache

router = APIRouter(prefix="/api/v1", tags=["inference"])

class GenerateRequest(BaseModel):
"""Generate request model."""

    model: str
    prompt: str
    cache_ttl: int = 3600

class GenerateResponse(BaseModel):
"""Generate response model."""

    text: str
    tokens: int
    cached: bool = False

# Global cache instance (initialize in app startup)

cache_manager: CacheManager | None = None
response_cache: ResponseCache | None = None

async def initialize_caches() -> None:
"""Initialize cache managers (call on app startup)."""
global cache_manager, response_cache

    cache_manager = CacheManager(redis_url="redis://redis:6379/0")
    await cache_manager.initialize()

    response_cache = ResponseCache(cache_manager, default_ttl=3600)

@router.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest) -> GenerateResponse:
"""Generate endpoint with caching and circuit breaker.

    1. Check cache for existing response
    2. If not cached, call model with circuit breaker
    3. Cache response for future requests
    4. Return response
    """
    if not response_cache:
        raise HTTPException(
            status_code=500,
            detail="Cache not initialized",
        )

    # Check cache first
    cached_response = await response_cache.get_response(
        model=request.model,
        prompt=request.prompt,
    )
    if cached_response:
        return GenerateResponse(
            **cached_response,
            cached=True,
        )

    # Not cached, call resilient client
    try:
        client = ResilientOllamaClient()
        text = await client.generate(request.model, request.prompt)

        response = GenerateResponse(
            text=text,
            tokens=len(text.split()),
            cached=False,
        )

        # Cache response
        await response_cache.set_response(
            model=request.model,
            prompt=request.prompt,
            response=response.model_dump(),
            ttl=request.cache_ttl,
        )

        return response

    except CircuitBreakerError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service temporarily unavailable: {e.message}",
        )

# Example 3: Circuit Breaker Monitoring Endpoint

# ===============================================

@router.get("/health/circuit-breakers")
async def get_circuit_breaker_status() -> dict:
"""Get status of all circuit breakers."""
manager = get_circuit_breaker_manager()
return {
"circuit_breakers": manager.get_state(),
"timestamp": **import**("datetime").datetime.utcnow().isoformat(),
}

# Example 4: Cache Metrics Endpoint

# ==================================

@router.get("/metrics/cache")
async def get_cache_metrics() -> dict:
"""Get cache metrics for monitoring."""
if not response_cache:
return {"error": "Cache not initialized"}

    return {
        "cache": response_cache.get_metrics(),
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
    }

# Example 5: Database Circuit Breaker Integration

# ================================================

from sqlalchemy.ext.asyncio import AsyncSession
from ollama.services.resilience import get_circuit_breaker_manager
from ollama.exceptions.circuit_breaker import CircuitBreakerError

class ResilientRepository:
"""Repository base class with circuit breaker protection."""

    def __init__(self, db: AsyncSession) -> None:
        """Initialize repository.

        Args:
            db: SQLAlchemy async session.
        """
        self.db = db
        manager = get_circuit_breaker_manager()
        self.breaker = manager.get_or_create(
            service_name="postgres",
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=2,
        )

    async def find_by_id(self, model_class: type, id: int):
        """Find record by ID with circuit breaker.

        Args:
            model_class: SQLAlchemy model class.
            id: Record ID.

        Returns:
            Record or None.

        Raises:
            CircuitBreakerError: If database is unavailable.
        """
        try:
            result = self.breaker.call(
                self.db.query(model_class).filter(model_class.id == id).first
            )
            return result
        except CircuitBreakerError:
            # Database is down, return None or cached fallback
            return None

# Example 6: Redis Circuit Breaker Integration

# =============================================

class ResilientCacheManager:
"""Cache manager with circuit breaker protection."""

    def __init__(self, redis_url: str = "redis://redis:6379/0"):
        """Initialize resilient cache.

        Args:
            redis_url: Redis connection URL.
        """
        self.cache = CacheManager(redis_url=redis_url)
        manager = get_circuit_breaker_manager()
        self.breaker = manager.get_or_create(
            service_name="redis",
            failure_threshold=10,
            recovery_timeout=30,
            success_threshold=3,
        )

    async def get(self, key: str):
        """Get value from cache with circuit breaker.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found or Redis unavailable.
        """
        try:
            return self.breaker.call(
                self.cache.get,
                key=key,
            )
        except CircuitBreakerError:
            # Redis is down, return None (graceful degradation)
            return None

    async def set(self, key: str, value, ttl: int | None = None) -> bool:
        """Set value in cache with circuit breaker.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds.

        Returns:
            True if successful, False otherwise.
        """
        try:
            return self.breaker.call(
                self.cache.set,
                key=key,
                value=value,
                ttl=ttl,
            )
        except CircuitBreakerError:
            # Redis is down, fail gracefully
            return False
