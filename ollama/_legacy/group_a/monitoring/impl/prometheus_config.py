"""
Prometheus Metrics Configuration
Configures Prometheus scrape targets and monitoring
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Prometheus configuration
PROMETHEUS_CONFIG: dict[str, Any] = {
    "global": {
        "scrape_interval": "15s",
        "evaluation_interval": "15s",
        "external_labels": {"cluster": "ollama-local", "environment": "development"},
    },
    "scrape_configs": [
        {"job_name": "prometheus", "static_configs": [{"targets": ["prometheus:9090"]}]},
        {
            "job_name": "ollama-api",
            "static_configs": [{"targets": ["ollama-api:8000"]}],
            "metrics_path": "/metrics",
        },
        {
            "job_name": "postgres",
            "static_configs": [{"targets": ["postgres-exporter:9187"]}],  # postgres_exporter
        },
    ],
    "alerting": {"alertmanagers": [{"static_configs": [{"targets": ["alertmanager:9093"]}]}]},
}

# Alert rules
ALERT_RULES: list[dict[str, Any]] = [
    {
        "alert": "HighErrorRate",
        "expr": "rate(http_requests_total{status_code=~'5..'}[5m]) > 0.1",
        "for": "5m",
        "annotations": {
            "summary": "High error rate detected",
            "description": "{{ $value }} errors per second in the last 5 minutes",
        },
    },
    {
        "alert": "HighLatency",
        "expr": "histogram_quantile(0.95, http_request_duration_seconds) > 1",
        "for": "5m",
        "annotations": {
            "summary": "High request latency detected",
            "description": "p95 latency is {{ $value }} seconds",
        },
    },
    {
        "alert": "OllamaUnavailable",
        "expr": "up{job='ollama'} == 0",
        "for": "1m",
        "annotations": {
            "summary": "Ollama inference engine is down",
            "description": "Ollama service has been unavailable for 1 minute",
        },
    },
    {
        "alert": "RateLimitExceeded",
        "expr": "rate(rate_limit_exceeded_total[5m]) > 10",
        "for": "5m",
        "annotations": {
            "summary": "High rate of rate limit exceeded events",
            "description": "{{ $value }} rate limit exceeded events per second",
        },
    },
    {
        "alert": "HighCacheEvictions",
        "expr": "rate(cache_evictions_total[5m]) > 100",
        "for": "5m",
        "annotations": {
            "summary": "High cache eviction rate",
            "description": "{{ $value }} cache evictions per second",
        },
    },
]


def get_prometheus_config() -> dict[str, Any]:
    """
    Get Prometheus configuration

    Returns:
        Prometheus config dictionary
    """
    return PROMETHEUS_CONFIG


def get_alert_rules() -> list[dict[str, Any]]:
    """
    Get Prometheus alert rules

    Returns:
        List of alert rules
    """
    return ALERT_RULES
