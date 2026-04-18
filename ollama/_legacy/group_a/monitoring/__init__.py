"""Monitoring domain."""

from .fastapi_instrumentation import OTLPInstrumentor
from .impl.jaeger_config import init_jaeger
from .impl.metrics import (
    AUTH_ATTEMPTS,
    CACHE_HITS,
    CACHE_MISSES,
    OLLAMA_TOKENS_GENERATED,
    RATE_LIMIT_EXCEEDED,
    REQUEST_COUNT,
    REQUEST_DURATION,
    REQUEST_SIZE,
    RESPONSE_SIZE,
    export_metrics,
    generate_latest,
    get_metrics_summary,
)
from .impl.metrics_middleware import MetricsCollectionMiddleware, setup_metrics_endpoints
from .otlp_collector import get_otlp_manager, setup_otlp_tracing

__all__ = [
    "AUTH_ATTEMPTS",
    "CACHE_HITS",
    "CACHE_MISSES",
    "OLLAMA_TOKENS_GENERATED",
    "RATE_LIMIT_EXCEEDED",
    "REQUEST_COUNT",
    "REQUEST_DURATION",
    "REQUEST_SIZE",
    "RESPONSE_SIZE",
    "MetricsCollectionMiddleware",
    "OTLPInstrumentor",
    "export_metrics",
    "generate_latest",
    "get_metrics_summary",
    "get_otlp_manager",
    "init_jaeger",
    "setup_metrics_endpoints",
    "setup_otlp_tracing",
]
