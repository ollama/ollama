"""
Grafana Dashboard Configuration
Provides pre-configured dashboards for monitoring
"""

from __future__ import annotations

from typing import Any

OLLAMA_API_DASHBOARD: dict[str, Any] = {
    "dashboard": {
        "title": "Ollama API Monitoring",
        "description": "Real-time monitoring of Ollama API performance and health",
        "tags": ["ollama", "api", "monitoring"],
        "timezone": "UTC",
        "panels": [
            {
                "id": 1,
                "title": "Request Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(http_requests_total[5m])",
                        "legendFormat": "{{ method }} {{ endpoint }}",
                    }
                ],
            },
            {
                "id": 2,
                "title": "Error Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(http_requests_total{status_code=~'5..'}[5m])",
                        "legendFormat": "{{ endpoint }}",
                    }
                ],
            },
            {
                "id": 3,
                "title": "Request Latency (p95)",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, http_request_duration_seconds)",
                        "legendFormat": "{{ method }} {{ endpoint }}",
                    }
                ],
            },
            {
                "id": 4,
                "title": "Ollama Generation Duration",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(ollama_generation_duration_seconds[5m]))",
                        "legendFormat": "{{ model }}",
                    }
                ],
            },
            {
                "id": 5,
                "title": "Tokens Generated",
                "type": "stat",
                "targets": [
                    {
                        "expr": "rate(ollama_tokens_generated_total[5m])",
                        "legendFormat": "{{ model }}",
                    }
                ],
            },
            {
                "id": 6,
                "title": "Cache Hit Rate",
                "type": "gauge",
                "targets": [
                    {
                        "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))",
                        "legendFormat": "{{ cache_type }}",
                    }
                ],
            },
            {
                "id": 7,
                "title": "Active Sessions",
                "type": "stat",
                "targets": [
                    {
                        "expr": "active_sessions",
                    }
                ],
            },
            {
                "id": 8,
                "title": "Rate Limit Exceeded Events",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(rate_limit_exceeded_total[5m])",
                        "legendFormat": "{{ endpoint }}",
                    }
                ],
            },
            {
                "id": 9,
                "title": "Database Connection Pool",
                "type": "stat",
                "targets": [
                    {
                        "expr": "db_connections_active / db_pool_size",
                        "legendFormat": "Pool Utilization",
                    }
                ],
            },
            {
                "id": 10,
                "title": "RAG Query Latency",
                "type": "heatmap",
                "targets": [
                    {"expr": "rag_query_latency_seconds_bucket", "legendFormat": "{{ le }}"}
                ],
            },
        ],
    }
}

SYSTEM_HEALTH_DASHBOARD: dict[str, Any] = {
    "dashboard": {
        "title": "System Health",
        "description": "System-wide health and resource monitoring",
        "tags": ["system", "health"],
        "panels": [
            {
                "id": 1,
                "title": "API Availability",
                "type": "stat",
                "targets": [
                    {
                        "expr": "up{job='ollama-api'}",
                    }
                ],
            },
            {
                "id": 2,
                "title": "Service Health Status",
                "type": "table",
                "targets": [
                    {
                        "expr": "up",
                    }
                ],
            },
            {
                "id": 3,
                "title": "Error Rate by Service",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(http_requests_total{status_code=~'5..'}[5m]) / rate(http_requests_total[5m])",
                        "legendFormat": "{{ job }}",
                    }
                ],
            },
            {
                "id": 4,
                "title": "P99 Latency",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds[5m]))",
                        "legendFormat": "{{ method }} {{ endpoint }}",
                    }
                ],
            },
        ],
    }
}


RESILIENCE_DASHBOARD: dict[str, Any] = {
    "dashboard": {
        "title": "Ollama System Resilience",
        "description": "Monitoring of circuit breakers, retries, and failure rates",
        "tags": ["ollama", "resilience", "circuit-breaker"],
        "timezone": "UTC",
        "panels": [
            {
                "id": 1,
                "title": "Circuit Breaker Status",
                "type": "state-timeline",
                "targets": [
                    {
                        "expr": "circuit_breaker_state",
                        "legendFormat": "{{ service }}",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "mappings": [
                            {"value": "0", "text": "CLOSED", "color": "green"},
                            {"value": "1", "text": "OPEN", "color": "red"},
                            {"value": "2", "text": "HALF-OPEN", "color": "yellow"},
                        ]
                    }
                },
            },
            {
                "id": 2,
                "title": "State Transitions",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(circuit_breaker_transitions_total[5m])",
                        "legendFormat": "{{ service }}: {{ from_state }} -> {{ to_state }}",
                    }
                ],
            },
            {
                "id": 3,
                "title": "Failure Rate (Circuit Breaker)",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(circuit_breaker_failures_total[5m])",
                        "legendFormat": "{{ service }} ({{ state }})",
                    }
                ],
            },
        ],
    }
}


def get_ollama_dashboard() -> dict[str, Any]:
    """Get Ollama API dashboard definition"""
    return OLLAMA_API_DASHBOARD


def get_system_health_dashboard() -> dict[str, Any]:
    """Get system health dashboard definition"""
    return SYSTEM_HEALTH_DASHBOARD


def get_resilience_dashboard() -> dict[str, Any]:
    """Get resilience dashboard definition"""
    return RESILIENCE_DASHBOARD
