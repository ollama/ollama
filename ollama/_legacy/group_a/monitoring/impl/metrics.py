"""
Metrics definitions for Ollama.

Provides Prometheus counters and histograms for request tracking,
response sizing, rate limiting, and cache metrics.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest

__all__ = [
    "AUTH_ATTEMPTS",
    "CACHE_HITS",
    "CACHE_MISSES",
    "CIRCUIT_BREAKER_FAILURES",
    "CIRCUIT_BREAKER_STATE",
    "CIRCUIT_BREAKER_TRANSITIONS",
    "OLLAMA_TOKENS_GENERATED",
    "RATE_LIMIT_EXCEEDED",
    "REQUEST_COUNT",
    "REQUEST_DURATION",
    "REQUEST_SIZE",
    "RESPONSE_SIZE",
    "export_metrics",
    "generate_latest",
    "get_metrics_summary",
]

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5],
)

REQUEST_SIZE = Histogram(
    "http_request_size_bytes",
    "Size of HTTP requests in bytes",
    ["method", "endpoint"],
    buckets=[500, 1_000, 5_000, 10_000, 50_000, 100_000],
)

RESPONSE_SIZE = Histogram(
    "http_response_size_bytes",
    "Size of HTTP responses in bytes",
    ["method", "endpoint", "status_code"],
    buckets=[500, 1_000, 5_000, 10_000, 50_000, 100_000],
)

RATE_LIMIT_EXCEEDED = Counter(
    "rate_limit_exceeded_total",
    "Total number of rate limit violations",
    ["endpoint"],
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_type", "operation"],
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_type", "operation"],
)

AUTH_ATTEMPTS = Counter(
    "auth_attempts_total",
    "Total authentication attempts",
    ["method", "result"],
)

OLLAMA_TOKENS_GENERATED = Counter(
    "ollama_tokens_generated_total",
    "Total tokens generated via Ollama inference",
    ["model", "type"],
)

CIRCUIT_BREAKER_FAILURES = Counter(
    "circuit_breaker_failures_total",
    "Total number of failures recorded by circuit breakers",
    ["service", "state"],
)

CIRCUIT_BREAKER_STATE = Gauge(
    "circuit_breaker_state",
    "Current state of the circuit breaker (0=closed, 1=open, 2=half_open)",
    ["service"],
)

CIRCUIT_BREAKER_TRANSITIONS = Counter(
    "circuit_breaker_transitions_total",
    "Total number of circuit breaker state transitions",
    ["service", "from_state", "to_state"],
)


def export_metrics() -> bytes:
    """Export all metrics in Prometheus format."""
    return generate_latest()


def _collect_counter_value(counter: Counter) -> float:
    """Return the sum of all samples for the provided counter."""

    total = 0.0
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name == counter._name:
                total += float(sample.value)
    return total


def get_metrics_summary() -> dict[str, float]:
    """Provide a simple summary of key counters for quick inspection."""

    return {
        "http_requests": _collect_counter_value(REQUEST_COUNT),
        "rate_limit_exceeded_total": _collect_counter_value(RATE_LIMIT_EXCEEDED),
        "cache_hits": _collect_counter_value(CACHE_HITS),
        "cache_misses": _collect_counter_value(CACHE_MISSES),
        "auth_attempts": _collect_counter_value(AUTH_ATTEMPTS),
        "ollama_tokens_generated": _collect_counter_value(OLLAMA_TOKENS_GENERATED),
    }
