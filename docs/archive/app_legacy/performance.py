"""
Performance optimization module with caching, database optimization, and compression.
"""

from typing import Any, Optional, Callable
from functools import wraps
import logging
import hashlib
import json
from datetime import datetime, timedelta
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
import gzip
import io

from app.core.redis_client import redis_client
from app.core.db import get_db

logger = logging.getLogger(__name__)


# ==================== Caching ====================

class CacheConfig:
    """Cache configuration."""
    DEFAULT_TTL = 3600  # 1 hour
    MAX_TTL = 86400  # 24 hours
    MIN_TTL = 60  # 1 minute


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_data = {
        "args": str(args),
        "kwargs": str(sorted(kwargs.items())),
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


async def get_cached(key: str) -> Optional[Any]:
    """Get value from cache."""
    try:
        cached = await redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.error(f"Cache get error: {e}")
    return None


async def set_cached(key: str, value: Any, ttl: int = CacheConfig.DEFAULT_TTL):
    """Set value in cache with TTL."""
    try:
        await redis_client.setex(
            key,
            ttl,
            json.dumps(value, default=str),
        )
    except Exception as e:
        logger.error(f"Cache set error: {e}")


async def invalidate_cache(pattern: str = "*"):
    """Invalidate cache by pattern."""
    try:
        keys = await redis_client.keys(pattern)
        if keys:
            await redis_client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache keys")
    except Exception as e:
        logger.error(f"Cache invalidation error: {e}")


def cache_response(ttl: int = CacheConfig.DEFAULT_TTL, namespace: str = "response"):
    """Decorator to cache endpoint responses."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{namespace}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_value = await get_cached(key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {key}")
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Cache result
            await set_cached(key, result, ttl)
            logger.debug(f"Cache miss: {key}")
            
            return result
        
        return wrapper
    
    return decorator


# ==================== Database Optimization ====================

class DatabaseOptimization:
    """Database optimization utilities."""
    
    @staticmethod
    async def create_indexes(db):
        """Create important database indexes."""
        indexes = [
            # Users table
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)",
            
            # Documents table
            "CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)",
            
            # Conversations table
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)",
            
            # Messages table
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)",
            
            # Embeddings table
            "CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings(vector)",
            
            # Usage table
            "CREATE INDEX IF NOT EXISTS idx_usage_user_id ON usage(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_usage_created_at ON usage(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_usage_model ON usage(model)",
            
            # Batch jobs table
            "CREATE INDEX IF NOT EXISTS idx_batch_jobs_user_id ON batch_jobs(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status)",
            "CREATE INDEX IF NOT EXISTS idx_batch_jobs_created_at ON batch_jobs(created_at)",
        ]
        
        try:
            for index_sql in indexes:
                await db.execute(index_sql)
            logger.info(f"Created {len(indexes)} database indexes")
        except Exception as e:
            logger.error(f"Index creation error: {e}")
    
    @staticmethod
    async def analyze_tables(db):
        """Analyze table statistics for query optimization."""
        tables = [
            "users", "documents", "conversations", "messages",
            "embeddings", "usage", "batch_jobs"
        ]
        
        try:
            for table in tables:
                await db.execute(f"ANALYZE {table}")
            logger.info(f"Analyzed {len(tables)} tables")
        except Exception as e:
            logger.error(f"Table analysis error: {e}")
    
    @staticmethod
    async def optimize_queries(db):
        """Run query optimization."""
        # Vacuum to clean up database
        try:
            await db.execute("VACUUM ANALYZE")
            logger.info("Database vacuum completed")
        except Exception as e:
            logger.error(f"Vacuum error: {e}")


# ==================== Connection Pooling ====================

class ConnectionPoolConfig:
    """Connection pool configuration."""
    POOL_SIZE = 20
    MAX_OVERFLOW = 10
    POOL_TIMEOUT = 30
    POOL_RECYCLE = 3600
    ECHO_POOL = False


class DatabaseConnectionPool:
    """Manages database connection pooling."""
    
    def __init__(self):
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "connection_errors": 0,
            "pool_exhausted_errors": 0,
        }
    
    async def get_connection_stats(self) -> dict:
        """Get connection pool statistics."""
        return self.stats.copy()
    
    @staticmethod
    def create_pool_config() -> dict:
        """Create SQLAlchemy pool configuration."""
        return {
            "poolclass": "sqlalchemy.pool.QueuePool",
            "pool_size": ConnectionPoolConfig.POOL_SIZE,
            "max_overflow": ConnectionPoolConfig.MAX_OVERFLOW,
            "pool_timeout": ConnectionPoolConfig.POOL_TIMEOUT,
            "pool_recycle": ConnectionPoolConfig.POOL_RECYCLE,
            "echo_pool": ConnectionPoolConfig.ECHO_POOL,
            "connect_args": {
                "connect_timeout": 10,
                "application_name": "ollama_api",
            }
        }


connection_pool = DatabaseConnectionPool()


# ==================== Response Compression ====================

class CompressionMiddleware:
    """Middleware for response compression (gzip)."""
    
    @staticmethod
    async def compress_response(response: Response) -> Response:
        """Compress response body if content-length > threshold."""
        MIN_COMPRESS_LENGTH = 1000  # 1KB
        
        # Skip if already compressed or streaming
        if isinstance(response, StreamingResponse):
            return response
        
        content_length = response.headers.get("content-length")
        if not content_length or int(content_length) < MIN_COMPRESS_LENGTH:
            return response
        
        # Get content
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Compress
        compressed = gzip.compress(body, compresslevel=6)
        
        # Return compressed response
        response.body = compressed
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = len(compressed)
        
        logger.debug(f"Compressed response: {len(body)} -> {len(compressed)} bytes")
        
        return response


# ==================== Query Optimization ====================

class QueryOptimizer:
    """Query optimization utilities."""
    
    @staticmethod
    def pagination_query(query, skip: int = 0, limit: int = 100):
        """Apply pagination to query."""
        return query.offset(skip).limit(limit)
    
    @staticmethod
    def select_specific_columns(query, columns: list):
        """Select only specific columns to reduce data transfer."""
        return query.with_entities(*columns)
    
    @staticmethod
    def eager_load_relations(query, *relations):
        """Use eager loading for relationships."""
        from sqlalchemy.orm import joinedload
        for relation in relations:
            query = query.options(joinedload(relation))
        return query
    
    @staticmethod
    async def batch_insert(db, model, items: list, batch_size: int = 1000):
        """Batch insert for better performance."""
        try:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                db.add_all(batch)
            await db.commit()
            logger.info(f"Batch inserted {len(items)} items")
        except Exception as e:
            await db.rollback()
            logger.error(f"Batch insert error: {e}")


# ==================== Load Testing Utilities ====================

class LoadTestingMetrics:
    """Metrics for load testing."""
    
    def __init__(self):
        self.request_times = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = None
        self.end_time = None
    
    def record_request(self, duration: float, error: bool = False):
        """Record request metrics."""
        self.request_times.append(duration)
        if error:
            self.error_count += 1
        else:
            self.success_count += 1
    
    def get_statistics(self) -> dict:
        """Get statistics."""
        if not self.request_times:
            return {}
        
        times = sorted(self.request_times)
        total = len(times)
        
        return {
            "total_requests": total,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "error_rate": (self.error_count / total * 100) if total > 0 else 0,
            "min_time_ms": round(times[0] * 1000, 2),
            "max_time_ms": round(times[-1] * 1000, 2),
            "mean_time_ms": round(sum(times) / total * 1000, 2),
            "median_time_ms": round(times[total // 2] * 1000, 2),
            "p95_time_ms": round(times[int(total * 0.95)] * 1000, 2),
            "p99_time_ms": round(times[int(total * 0.99)] * 1000, 2),
            "throughput_rps": total / (self.end_time - self.start_time) if self.start_time and self.end_time else 0,
        }


# ==================== Performance Middleware ====================

class PerformanceMonitoringMiddleware:
    """Middleware for performance monitoring."""
    
    def __init__(self, app):
        self.app = app
        self.metrics = LoadTestingMetrics()
    
    async def __call__(self, request: Request) -> Response:
        """Monitor request performance."""
        import time
        
        start_time = time.time()
        
        try:
            response = await self.app(request)
            duration = time.time() - start_time
            
            # Record metrics
            self.metrics.record_request(duration, error=False)
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(duration)
            response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
            
            if duration > 1.0:
                logger.warning(f"Slow request: {request.method} {request.url.path} ({duration:.2f}s)")
            
            return response
        
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_request(duration, error=True)
            logger.error(f"Request error: {request.method} {request.url.path} ({e})")
            raise


# ==================== Health Check Optimization ====================

async def health_check_with_cache():
    """Health check with caching."""
    cache_key = "health_check:status"
    
    # Try to get from cache
    cached_status = await get_cached(cache_key)
    if cached_status is not None:
        return cached_status
    
    # Perform health checks
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": await check_database_health(),
        "cache": await check_cache_health(),
    }
    
    # Cache for 30 seconds
    await set_cached(cache_key, status, ttl=30)
    
    return status


async def check_database_health() -> dict:
    """Check database connectivity."""
    try:
        db = get_db()
        # Simple query to verify connection
        await db.execute("SELECT 1")
        return {"status": "healthy", "response_time_ms": 10}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_cache_health() -> dict:
    """Check cache connectivity."""
    try:
        test_key = "health:check:test"
        await redis_client.set(test_key, "ok", ex=10)
        value = await redis_client.get(test_key)
        if value == b"ok":
            return {"status": "healthy"}
        return {"status": "unhealthy", "error": "Cache read/write failed"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# ==================== Rate Limiting ====================

class RateLimiter:
    """Rate limiting using Redis."""
    
    @staticmethod
    async def check_rate_limit(
        key: str,
        limit: int = 100,
        window: int = 60,
    ) -> tuple[bool, dict]:
        """
        Check if request is within rate limit.
        
        Returns: (is_allowed, info)
        """
        try:
            current = await redis_client.incr(key)
            
            if current == 1:
                # First request in window, set expiry
                await redis_client.expire(key, window)
            
            is_allowed = current <= limit
            remaining = max(0, limit - current)
            
            return is_allowed, {
                "limit": limit,
                "remaining": remaining,
                "reset_in_seconds": await redis_client.ttl(key),
            }
        
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request if cache is down
            return True, {"error": str(e)}


import asyncio
