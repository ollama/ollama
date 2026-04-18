"""
Ollama API - Main Application Entry Point
FastAPI-based AI inference server with production-grade features
"""

import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Phase 6: Error handling and rate limiting
from ollama.api.error_handlers import register_exception_handlers
from ollama.api.routes import (
    auth,
    conversations,
    documents,
    health,
    inference,
    usage,
)

# Phase 8: Consolidated settings (replaces old config)
from ollama.config.settings import Settings, get_settings
from ollama.middleware.impl.rate_limit import RateLimitMiddleware

# Phase 6: Rate limiting
from ollama.middleware.rate_limiter import RateLimiter

# Phase 7: Performance monitoring
from ollama.monitoring import (  # type: ignore[attr-defined]
    MetricsCollectionMiddleware,
    OTLPInstrumentor,
    setup_metrics_endpoints,
    setup_otlp_tracing,
)
from ollama.monitoring.profiling_endpoint import router as profiling_router
from ollama.services import get_db_manager, init_cache, init_database, init_vector_db
from ollama.services.inference.ollama_client import (
    clear_ollama_client,
    get_ollama_client,
    init_ollama_client,
)
from ollama.services.models.vector import VectorManager
from ollama.services.persistence.cache import CacheManager
from ollama.services.resources.manager import ResourceManager
from ollama.training.routes import jobs as training_jobs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Global manager instances - MUST be initialized before use
_cache_manager: CacheManager | None = None
_vector_manager: VectorManager | None = None
_resource_manager: ResourceManager | None = None
_training_worker: Any = None


async def get_cache_manager() -> CacheManager:
    """Get cache manager instance.

    Returns:
        Initialized CacheManager instance

    Raises:
        RuntimeError: If cache manager not initialized
    """
    global _cache_manager
    if _cache_manager is None:
        msg = "Cache manager not initialized. Call init_cache() during startup."
        raise RuntimeError(msg)
    return _cache_manager


async def get_vector_manager() -> VectorManager:
    """Get vector manager instance.

    Returns:
        Initialized VectorManager instance

    Raises:
        RuntimeError: If vector manager not initialized
    """
    global _vector_manager
    if _vector_manager is None:
        msg = "Vector manager not initialized. Call init_vector_db() during startup."
        raise RuntimeError(msg)
    return _vector_manager


async def get_resource_manager() -> ResourceManager:
    """Get resource manager instance.

    Returns:
        Initialized ResourceManager instance

    Raises:
        RuntimeError: If resource manager not initialized
    """
    global _resource_manager
    if _resource_manager is None:
        msg = "Resource manager not initialized. Call init_resources() during startup."
        raise RuntimeError(msg)
    return _resource_manager


async def get_training_engine() -> Any | None:
    """Get training engine instance from worker.

    Returns:
        Training engine instance or None if not available
    """
    global _training_worker
    if _training_worker is None:
        return None
    return _training_worker.engine


async def _startup_firebase(settings: Settings) -> None:
    """Initialize Firebase OAuth if enabled."""
    firebase_enabled = getattr(settings, "firebase_enabled", False)
    if firebase_enabled:
        logger.info("🔐 Initializing Firebase OAuth...")
        try:
            from ollama.auth import init_firebase

            firebase_creds = getattr(settings, "firebase_credentials_path", None)
            init_firebase(firebase_creds)
            logger.info("✅ Firebase OAuth initialized")
        except Exception as e:
            logger.warning(f"⚠️  Firebase OAuth initialization failed: {e}")
            logger.warning("⚠️  Protected endpoints will return 503 Service Unavailable")
    else:
        logger.info("⚠️  Firebase OAuth disabled (FIREBASE_ENABLED=false)")


async def _startup_database(settings: Settings) -> None:
    """Initialize database connection pool."""
    logger.info("📦 Initializing database connection...")
    try:
        # Phase 8: Use consolidated settings with auto-generated URL
        db_url = str(settings.database.url)
        db_manager = init_database(db_url, echo=False)
        await db_manager.initialize()
        logger.info("✅ Database connected")
    except Exception as e:
        logger.warning(f"⚠️  Database unavailable: {e}")
        logger.warning("⚠️  Running in degraded mode - persistence disabled")


async def _startup_cache(settings: Settings) -> None:
    """Initialize Redis connection."""
    logger.info("🔴 Connecting to Redis...")
    try:
        from ollama.api.dependencies.cache import set_global_cache_manager
        from ollama.services.cache.resilient_cache import ResilientCacheManager

        # Phase 8: Use consolidated settings with auto-generated URL
        redis_url = str(settings.redis.url)
        redis_db = settings.redis.db
        cache_manager = init_cache(redis_url, db=redis_db)
        await cache_manager.initialize()

        # Wrap in resilient manager
        resilient_cache = ResilientCacheManager(cache_manager)
        set_global_cache_manager(resilient_cache)

        logger.info("✅ Redis connected (Resilience enabled)")
    except Exception as e:
        logger.warning(f"⚠️  Redis unavailable: {e}")
        logger.warning("⚠️  Caching disabled")


async def _startup_vector_db(settings: Settings) -> None:
    """Initialize Qdrant client."""
    logger.info("🔷 Connecting to Qdrant...")
    try:
        from ollama.api.dependencies.vector import set_global_vector_manager
        from ollama.services.models.resilient_vector import ResilientVectorManager

        # Phase 8: Use consolidated settings with auto-generated URL
        vector_url = str(settings.vector_db.url)

        vector_manager = init_vector_db(vector_url)
        await vector_manager.initialize()

        # Wrap in resilient manager
        resilient_vector = ResilientVectorManager(vector_manager)
        set_global_vector_manager(resilient_vector)

        logger.info("✅ Qdrant connected (Resilience enabled)")
    except Exception as e:
        logger.warning(f"⚠️  Qdrant unavailable: {e}")
        logger.warning("⚠️  Vector search disabled")


async def _startup_ollama(settings: Settings) -> None:
    """Initialize Ollama inference client."""
    logger.info("🤖 Connecting to Ollama inference engine...")
    # Phase 8: Use consolidated settings
    ollama_base_url = str(settings.ollama.base_url)
    ollama_timeout = float(settings.ollama.timeout)

    ollama_client = init_ollama_client(
        base_url=ollama_base_url,
        timeout=ollama_timeout,
    )
    try:
        await ollama_client.initialize()
        logger.info("✅ Ollama inference engine connected")
    except Exception as e:
        clear_ollama_client()
        logger.warning(f"⚠️  Ollama inference engine not available: {e}")
        logger.warning("⚠️  API will return stub responses for model operations")


async def _startup_training_worker() -> None:
    """Initialize and start the training background worker."""
    logger.info("🏋️ Initializing training worker...")
    try:
        from pathlib import Path

        from ollama.main import get_resource_manager
        from ollama.services.persistence.database import get_db_manager
        from ollama.training.services.engine import TrainingEngine
        from ollama.training.services.worker import TrainingWorker

        db_manager = get_db_manager()
        resource_manager = await get_resource_manager()

        # FAANG-Grade Architecture: The engine configuration should come from settings
        # Local dev defaults:
        base_model_path = Path("/home/akushnir/ollama/models/base")
        output_dir = Path("/home/akushnir/ollama/models/trained")

        engine = TrainingEngine(base_model_path=base_model_path, output_dir=output_dir)
        worker = TrainingWorker(
            db_manager=db_manager,
            engine=engine,
            resources=resource_manager,
            poll_interval=10.0,
        )

        global _training_worker
        _training_worker = worker
        await worker.start()
        logger.info("✅ Training worker started")

    except Exception as e:
        logger.error(f"❌ Failed to start training worker: {e}")
        # We don't raise here as training is an enhancement, not core to inference


async def _startup_tracing(app: FastAPI) -> None:
    """Initialize OpenTelemetry distributed tracing via OTLP."""
    logger.info("🔍 Initializing OpenTelemetry distributed tracing (OTLP)...")
    try:
        settings = get_settings()
        # Initialize OTLP Collector Manager
        setup_otlp_tracing(
            service_name="ollama-api",
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"),
            sample_rate=1.0 if settings.debug else 0.1,
        )

        # Instrument FastAPI and other libraries
        OTLPInstrumentor.instrument_all(app=app)

        logger.info("✅ OpenTelemetry tracing initialized (OTLP)")
    except Exception as e:
        logger.warning(f"⚠️  OpenTelemetry tracing not available: {e}")


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown."""
    settings = get_settings()
    logger.info("🚀 Starting Ollama API Server")

    # Initialize tracing
    await _startup_tracing(app)
    # Phase 8: Environment detection
    env = settings.environment.value if hasattr(settings, "environment") else "production"
    logger.info(f"Environment: {env}")
    # Phase 8: API settings
    if hasattr(settings, "api"):
        logger.info(f"Host: {settings.api.host}:{settings.api.port}")
    else:
        host = getattr(settings, "host", "0.0.0.0")
        port = getattr(settings, "port", 8000)
        logger.info(f"Host: {host}:{port}")
    public_url = getattr(settings, "public_url", "http://localhost:8000")
    logger.info(f"Public URL: {public_url}")

    # Startup tasks
    try:
        await _startup_firebase(settings)
        await _startup_database(settings)
        await _startup_cache(settings)
        await _startup_vector_db(settings)
        await _startup_ollama(settings)
        await _startup_training_worker()
        logger.info("✅ Ollama API Server started successfully")

    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        raise

    yield

    # Shutdown tasks
    logger.info("🛑 Shutting down Ollama API Server")
    try:
        # Stop training worker
        global _training_worker
        if _training_worker:
            await _training_worker.stop()
            logger.info("✅ Training worker stopped")

        # Close Ollama client
        try:
            ollama_client = get_ollama_client()
            await ollama_client.close()
        except RuntimeError:
            pass

        # Close database connection
        db_manager = get_db_manager()
        await db_manager.close()

        # Close Redis connection
        cache_manager = await get_cache_manager()
        await cache_manager.close()

        # Close Qdrant connection
        vector_manager = await get_vector_manager()
        await vector_manager.close()

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    logger.info("✅ Shutdown complete")


# Create FastAPI application
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()

    # Initialize resource manager
    global _resource_manager
    _resource_manager = ResourceManager()

    app = FastAPI(
        title="Ollama API",
        description="Elite AI Inference Platform - Local LLM serving with production features",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Phase 6: Register exception handlers for structured error responses
    register_exception_handlers(app)
    logger.info("✅ Phase 6: Exception handlers registered")

    # Phase 6: Initialize rate limiter with Redis backend
    try:
        redis_url = str(settings.redis.url)
        app.state.rate_limiter = RateLimiter(redis_url=redis_url)
        logger.info("✅ Phase 6: Rate limiter initialized (Redis backend)")
    except Exception as e:
        # Fallback to in-memory rate limiter
        app.state.rate_limiter = RateLimiter()
        logger.warning(f"⚠️  Phase 6: Rate limiter using in-memory fallback: {e}")

    # CORS Middleware
    # Phase 8: Use consolidated API settings
    if hasattr(settings, "api"):
        cors_origins = settings.api.cors_origins
    else:
        cors_origins = getattr(settings, "cors_origins", ["*"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=getattr(settings, "cors_allow_credentials", True),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=getattr(settings, "cors_expose_headers", []),
    )

    # Metrics collection middleware (early in stack for accurate timing)
    app.add_middleware(MetricsCollectionMiddleware)

    # Response Caching Middleware (add after CORS)
    # Note: This will be added dynamically in create_app after cache_manager is available
    # For now, we'll add a hook to set it up after initialization

    # Rate limiting middleware (add before other middleware)
    # Phase 8: Use consolidated API settings
    if hasattr(settings, "api"):
        rate_limit_per_minute = settings.api.rate_limit_requests
        rate_limit_burst = getattr(settings.api, "rate_limit_burst", 100)
    else:
        rate_limit_per_minute = getattr(settings, "rate_limit_per_minute", 60)
        rate_limit_burst = getattr(settings, "rate_limit_burst", 100)

    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=rate_limit_per_minute,
        burst_size=rate_limit_burst,
        exclude_paths=["/health", "/docs", "/openapi.json", "/metrics"],
    )

    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Trusted hosts (when behind reverse proxy)
    trusted_hosts = getattr(settings, "trusted_hosts", None)
    if trusted_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID", "no-request-id")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    # Phase 6: Exception handlers are now registered via register_exception_handlers()
    # This provides structured error responses with request_id and timestamp

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
    # Centralized inference router (covers models, generate, chat, embeddings)
    app.include_router(inference.router)
    app.include_router(conversations.router, tags=["Conversations"])
    app.include_router(documents.router, tags=["Documents"])
    app.include_router(usage.router, tags=["Usage"])
    app.include_router(training_jobs.router, prefix="/api/v1/training", tags=["Training"])

    # Development-only profiling endpoints (do not expose in production)
    try:
        app.include_router(profiling_router, prefix="/monitoring", tags=["Profiling"])
    except Exception:
        # Defensive: if instrumentation/profiling unavailable, continue
        logger.debug("Profiling router not available; skipping")

    # Setup metrics endpoints
    setup_metrics_endpoints(app)

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": "Ollama API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create application instance
app = create_app()


def main() -> None:
    """Run the application with uvicorn"""
    settings = get_settings()

    # Phase 8: Use consolidated API and monitoring settings
    if hasattr(settings, "api"):
        host = settings.api.host
        port = settings.api.port
        workers = settings.api.workers
        reload = settings.api.reload
    else:
        host = getattr(settings, "host", "0.0.0.0")
        port = getattr(settings, "port", 8000)
        workers = getattr(settings, "workers", 4)
        reload = False

    if hasattr(settings, "monitoring"):
        log_level = settings.monitoring.log_level.lower()
    else:
        log_level = getattr(settings, "log_level", "info").lower()

    uvicorn.run(
        "ollama.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
