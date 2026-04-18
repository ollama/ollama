"""Prometheus dashboard configuration and monitoring setup.

Defines default dashboards, alert rules, and monitoring
targets for Ollama infrastructure.
"""

import json
from typing import Any


def get_ollama_dashboard_json() -> str:
    """Get Prometheus dashboard JSON for Ollama.

    Returns:
        Grafana dashboard JSON definition
    """
    dashboard = {
        "dashboard": {
            "title": "Ollama API Monitoring",
            "panels": [
                {
                    "title": "Request Rate (Requests/sec)",
                    "targets": [
                        {
                            "expr": "rate(ollama_api_requests_total[5m])",
                        }
                    ],
                    "type": "graph",
                },
                {
                    "title": "Request Latency P95 (ms)",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(ollama_inference_latency_seconds_bucket[5m])) * 1000",
                        }
                    ],
                    "type": "graph",
                },
                {
                    "title": "Request Latency P99 (ms)",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.99, rate(ollama_inference_latency_seconds_bucket[5m])) * 1000",
                        }
                    ],
                    "type": "graph",
                },
                {
                    "title": "Error Rate",
                    "targets": [
                        {
                            "expr": 'rate(ollama_api_requests_total{status=~"5.."}[5m]) / rate(ollama_api_requests_total[5m])',
                        }
                    ],
                    "type": "graph",
                },
                {
                    "title": "Cache Hit Rate",
                    "targets": [
                        {
                            "expr": "rate(ollama_cache_hits_total[5m]) / rate(ollama_cache_requests_total[5m])",
                        }
                    ],
                    "type": "graph",
                },
                {
                    "title": "Model Load Time (seconds)",
                    "targets": [
                        {
                            "expr": "ollama_model_load_duration_seconds",
                        }
                    ],
                    "type": "graph",
                },
                {
                    "title": "Inference Tokens/second",
                    "targets": [
                        {
                            "expr": "rate(ollama_inference_tokens_total[5m])",
                        }
                    ],
                    "type": "graph",
                },
                {
                    "title": "Active Connections",
                    "targets": [
                        {
                            "expr": "ollama_active_connections",
                        }
                    ],
                    "type": "stat",
                },
                {
                    "title": "Memory Usage (MB)",
                    "targets": [
                        {
                            "expr": 'container_memory_usage_bytes{container="ollama"} / 1024 / 1024',
                        }
                    ],
                    "type": "graph",
                },
                {
                    "title": "CPU Usage (%)",
                    "targets": [
                        {
                            "expr": 'rate(container_cpu_usage_seconds_total{container="ollama"}[5m]) * 100',
                        }
                    ],
                    "type": "graph",
                },
            ],
            "uid": "ollama",
            "version": 1,
        }
    }

    return json.dumps(dashboard)


def get_alert_rules() -> str:
    """Get Prometheus alert rules for Ollama.

    Returns:
        YAML-formatted alert rules
    """
    rules = """groups:
  - name: ollama_alerts
    interval: 30s
    rules:
      # Request latency alerts
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, rate(ollama_inference_latency_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency"
          description: "P95 inference latency is {{ $value }}s"

      - alert: CriticalInferenceLatency
        expr: histogram_quantile(0.99, rate(ollama_inference_latency_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Critical inference latency"
          description: "P99 inference latency is {{ $value }}s"

      # Error rate alerts
      - alert: HighErrorRate
        expr: |
          rate(ollama_api_requests_total{status=~"5.."}[5m]) /
          rate(ollama_api_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: CriticalErrorRate
        expr: |
          rate(ollama_api_requests_total{status=~"5.."}[5m]) /
          rate(ollama_api_requests_total[5m]) > 0.10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Critical API error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # Cache hit rate alert
      - alert: LowCacheHitRate
        expr: |
          rate(ollama_cache_hits_total[5m]) /
          rate(ollama_cache_requests_total[5m]) < 0.70
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value | humanizePercentage }}"

      # Resource utilization alerts
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes{container="ollama"} / container_spec_memory_limit_bytes > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      - alert: HighCPUUsage
        expr: rate(container_cpu_usage_seconds_total{container="ollama"}[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value | humanizePercentage }}"

      # Database alerts
      - alert: DatabaseConnectionPoolExhausted
        expr: ollama_db_connections_used / ollama_db_connections_limit > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "{{ $value | humanizePercentage }} of connections in use"

      # Model loading alerts
      - alert: ModelLoadFailure
        expr: increase(ollama_model_load_errors_total[5m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model load failure"
          description: "{{ $value }} model load errors in last 5 minutes"

      # Rate limiting alerts
      - alert: HighRateLimitErrors
        expr: |
          rate(ollama_api_requests_total{status="429"}[5m]) /
          rate(ollama_api_requests_total[5m]) > 0.10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate limiting errors"
          description: "{{ $value | humanizePercentage }} of requests are rate limited"
"""

    return rules


def get_slo_definitions() -> dict[str, Any]:
    """Get SLO definitions for Ollama.

    Returns:
        Dictionary of SLO names to thresholds
    """
    return {
        "inference_latency_p95_ms": 500,
        "inference_latency_p99_ms": 2000,
        "api_response_time_p95_ms": 200,
        "api_response_time_p99_ms": 500,
        "error_rate_max_percent": 1.0,
        "cache_hit_rate_min_percent": 70.0,
        "availability_percent": 99.5,
        "model_load_time_max_seconds": 30,
        "token_throughput_min_per_sec": 50,
    }


def get_recording_rules() -> str:
    """Get Prometheus recording rules for Ollama.

    Returns:
        YAML-formatted recording rules
    """
    rules = """groups:
  - name: ollama_recording
    interval: 30s
    rules:
      # Inference metrics
      - record: ollama:inference_latency:p50
        expr: histogram_quantile(0.50, rate(ollama_inference_latency_seconds_bucket[5m]))

      - record: ollama:inference_latency:p95
        expr: histogram_quantile(0.95, rate(ollama_inference_latency_seconds_bucket[5m]))

      - record: ollama:inference_latency:p99
        expr: histogram_quantile(0.99, rate(ollama_inference_latency_seconds_bucket[5m]))

      # Request metrics
      - record: ollama:api:request_rate
        expr: rate(ollama_api_requests_total[5m])

      - record: ollama:api:error_rate
        expr: rate(ollama_api_requests_total{status=~"5.."}[5m])

      # Cache metrics
      - record: ollama:cache:hit_rate
        expr: |
          rate(ollama_cache_hits_total[5m]) /
          rate(ollama_cache_requests_total[5m])

      # Resource metrics
      - record: ollama:memory:usage_percent
        expr: |
          container_memory_usage_bytes{container="ollama"} /
          container_spec_memory_limit_bytes * 100

      - record: ollama:cpu:usage_percent
        expr: rate(container_cpu_usage_seconds_total{container="ollama"}[5m]) * 100
"""

    return rules


def get_scrape_config() -> dict[str, Any]:
    """Get Prometheus scrape configuration.

    Returns:
        Scrape configuration dictionary
    """
    return {
        "global": {
            "scrape_interval": "30s",
            "evaluation_interval": "30s",
            "external_labels": {
                "cluster": "ollama-production",
                "environment": "production",
            },
        },
        "scrape_configs": [
            {
                "job_name": "ollama-api",
                "static_configs": [
                    {
                        "targets": ["localhost:8000"],
                    }
                ],
                "metrics_path": "/metrics",
            },
            {
                "job_name": "postgres",
                "static_configs": [
                    {
                        "targets": ["postgres:9187"],
                    }
                ],
            },
            {
                "job_name": "redis",
                "static_configs": [
                    {
                        "targets": ["redis:9121"],
                    }
                ],
            },
            {
                "job_name": "docker",
                "static_configs": [
                    {
                        "targets": ["localhost:9323"],
                    }
                ],
            },
        ],
    }
