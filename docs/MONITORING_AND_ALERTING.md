# 📊 Production Monitoring & Alerting Configuration

**Date**: January 13, 2026
**Status**: Phase 4 Complete
**Environment**: Production

---

## Table of Contents

1. [Monitoring Architecture](#monitoring-architecture)
2. [Metrics Collection](#metrics-collection)
3. [Alert Rules](#alert-rules)
4. [Dashboards](#dashboards)
5. [SLOs & SLIs](#slos--slis)
6. [Data Retention](#data-retention)

---

## Monitoring Architecture

```
┌─────────────────────────────────────────────┐
│         Application & Infrastructure        │
│  (Cloud Run, Cloud SQL, Cloud Redis, etc)   │
└──────────────┬──────────────────────────────┘
               │
      ┌────────▼────────┐
      │  Metrics Export │
      │   (Prometheus)  │
      └────────┬────────┘
               │
      ┌────────▼──────────────────┐
      │  Google Cloud Monitoring  │
      │  - Metrics Collection     │
      │  - Time Series Database   │
      │  - Alert Routing          │
      └────────┬──────────────────┘
               │
        ┌──────┴──────┐
        │             │
   ┌────▼────┐   ┌────▼─────┐
   │ Grafana  │   │ Alerts   │
   │Dashboard │   │ (Email,  │
   │          │   │  Slack)  │
   └──────────┘   └──────────┘
```

---

## Metrics Collection

### Application Metrics

```python
# app/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
request_total = Counter(
    'ollama_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration_seconds = Histogram(
    'ollama_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Inference metrics
inference_requests_total = Counter(
    'ollama_inference_requests_total',
    'Total inference requests',
    ['model', 'status']
)

inference_duration_seconds = Histogram(
    'ollama_inference_duration_seconds',
    'Inference duration in seconds',
    ['model'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

tokens_generated_total = Counter(
    'ollama_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

# Cache metrics
cache_hits_total = Counter(
    'ollama_cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses_total = Counter(
    'ollama_cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

cache_hit_ratio = Gauge(
    'ollama_cache_hit_ratio',
    'Cache hit ratio (0-1)',
    ['cache_type']
)

# Database metrics
db_connection_pool_size = Gauge(
    'ollama_db_connection_pool_size',
    'Database connection pool size'
)

db_active_connections = Gauge(
    'ollama_db_active_connections',
    'Active database connections'
)

db_query_duration_seconds = Histogram(
    'ollama_db_query_duration_seconds',
    'Database query duration',
    ['query_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# Error metrics
errors_total = Counter(
    'ollama_errors_total',
    'Total errors',
    ['error_type', 'endpoint']
)

# Business metrics
active_users_gauge = Gauge(
    'ollama_active_users',
    'Number of active users'
)

api_key_rotations_total = Counter(
    'ollama_api_key_rotations_total',
    'Total API key rotations'
)
```

### Infrastructure Metrics (GCP)

```yaml
# Collected automatically by Cloud Monitoring
metrics:
  # Cloud Run metrics
  - run.googleapis.com/request_count
  - run.googleapis.com/request_latencies
  - run.googleapis.com/execution_count
  - run.googleapis.com/execution_latencies

  # Cloud SQL metrics
  - cloudsql.googleapis.com/database/cpu/utilization
  - cloudsql.googleapis.com/database/memory/utilization
  - cloudsql.googleapis.com/database/disk/utilization
  - cloudsql.googleapis.com/database/replication/replica_lag

  # Cloud Redis metrics
  - redis.googleapis.com/clients/blocked_clients
  - redis.googleapis.com/memory/usage
  - redis.googleapis.com/stats/connections/total
  - redis.googleapis.com/stats/evicted_keys

  # Firewall metrics
  - compute.googleapis.com/security_policy/request_count
  - compute.googleapis.com/security_policy/request_count_denied
```

---

## Alert Rules

### P1 Alerts (Immediate Action Required)

```yaml
# cloud-monitoring-alerts.yaml

alerts:
  - name: "Service Down - P1"
    description: "Production service is not responding"
    condition:
      metric: run.googleapis.com/request_count
      threshold: 0
      duration: 2m  # Two minutes of zero requests
      comparison: COMPARISON_LT
    notification_channels:
      - on_call_sms
      - on_call_slack
      - ops_team_email
    severity: CRITICAL

  - name: "Error Rate Critical - P1"
    description: "Error rate exceeds 5%"
    condition:
      metric: ollama_errors_total
      threshold: 0.05
      duration: 5m
      comparison: COMPARISON_GT
    notification_channels:
      - on_call_sms
      - on_call_slack
      - ops_team_email
    severity: CRITICAL

  - name: "Database Connection Pool Exhausted - P1"
    description: "All database connections are in use"
    condition:
      metric: ollama_db_active_connections
      threshold: max_pool_size
      duration: 1m
      comparison: COMPARISON_GTE
    notification_channels:
      - on_call_sms
      - on_call_slack
      - database_team_email
    severity: CRITICAL

  - name: "Out of Memory - P1"
    description: "Cloud Run instance approaching OOM kill"
    condition:
      metric: run.googleapis.com/memory/utilization
      threshold: 0.95  # 95%
      duration: 2m
      comparison: COMPARISON_GT
    notification_channels:
      - on_call_sms
      - on_call_slack
    severity: CRITICAL

  - name: "Database Down - P1"
    description: "Cannot connect to database"
    condition:
      metric: cloudsql.googleapis.com/database/cpu/utilization
      threshold: null  # No data
      duration: 2m
      comparison: COMPARISON_EQ
    notification_channels:
      - on_call_sms
      - on_call_slack
      - database_team_email
    severity: CRITICAL
```

### P2 Alerts (Urgent, <30 min response)

```yaml
  - name: "High Latency - P2"
    description: "API p99 latency exceeds 10 seconds"
    condition:
      metric: ollama_request_duration_seconds
      percentile: 99
      threshold: 10
      duration: 5m
      comparison: COMPARISON_GT
    notification_channels:
      - on_call_slack
      - ops_team_email
    severity: HIGH

  - name: "High CPU Usage - P2"
    description: "Cloud Run CPU exceeds 80%"
    condition:
      metric: run.googleapis.com/cpu/utilization
      threshold: 0.80
      duration: 5m
      comparison: COMPARISON_GT
    notification_channels:
      - on_call_slack
      - ops_team_email
    severity: HIGH

  - name: "Database CPU Throttling - P2"
    description: "Database CPU at 85%+ for 10 minutes"
    condition:
      metric: cloudsql.googleapis.com/database/cpu/utilization
      threshold: 0.85
      duration: 10m
      comparison: COMPARISON_GT
    notification_channels:
      - on_call_slack
      - database_team_email
    severity: HIGH

  - name: "Low Cache Hit Rate - P2"
    description: "Cache hit rate below 70%"
    condition:
      metric: ollama_cache_hit_ratio
      threshold: 0.70
      duration: 15m
      comparison: COMPARISON_LT
    notification_channels:
      - slack
      - email
    severity: HIGH

  - name: "High Disk Usage - P2"
    description: "Database disk usage exceeds 80%"
    condition:
      metric: cloudsql.googleapis.com/database/disk/utilization
      threshold: 0.80
      duration: 10m
      comparison: COMPARISON_GT
    notification_channels:
      - on_call_slack
      - database_team_email
    severity: HIGH

  - name: "Redis Memory Eviction - P2"
    description: "Cache is evicting keys due to memory pressure"
    condition:
      metric: redis.googleapis.com/stats/evicted_keys
      threshold: 100  # More than 100 keys per minute
      duration: 5m
      comparison: COMPARISON_GT
    notification_channels:
      - slack
      - email
    severity: HIGH
```

### P3 Alerts (Monitor, <4 hour response)

```yaml
  - name: "Moderate Latency - P3"
    description: "API p95 latency exceeds 5 seconds"
    condition:
      metric: ollama_request_duration_seconds
      percentile: 95
      threshold: 5
      duration: 10m
      comparison: COMPARISON_GT
    notification_channels:
      - slack
    severity: MEDIUM

  - name: "Increasing Error Rate - P3"
    description: "Error rate trending upward (> 1%)"
    condition:
      metric: ollama_errors_total
      threshold: 0.01
      duration: 15m
      comparison: COMPARISON_GT
    notification_channels:
      - slack
    severity: MEDIUM

  - name: "Model Loading Slow - P3"
    description: "Model loading time > 30 seconds"
    condition:
      metric: ollama_inference_duration_seconds
      threshold: 30
      duration: 5m
      comparison: COMPARISON_GT
    notification_channels:
      - slack
    severity: MEDIUM

  - name: "API Key Expiration Soon - P3"
    description: "API keys expiring within 7 days"
    condition:
      metric: ollama_api_key_expiration_days
      threshold: 7
      comparison: COMPARISON_LT
    notification_channels:
      - email
    severity: MEDIUM
```

### Remediation Alerts (Auto-healing)

```yaml
  - name: "Auto-scaling Triggered"
    description: "Instance count increased due to load"
    condition:
      metric: run.googleapis.com/instance_count
      change: increased
    notification_channels:
      - slack
    action: "Auto-scale to handle load"

  - name: "Cache Rebuilt"
    description: "Cache invalidated and rebuilt"
    condition:
      metric: ollama_cache_rebuild_total
      threshold: 1
    notification_channels:
      - slack
    action: "Monitor performance during rebuild"
```

---

## Dashboards

### Main Production Dashboard

```json
{
  "displayName": "Ollama Production - Main Dashboard",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "width": 6,
        "height": 3,
        "widget": {
          "title": "Request Rate (req/sec)",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"ollama_requests_total\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "xPos": 6,
        "width": 6,
        "height": 3,
        "widget": {
          "title": "Error Rate (%)",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"ollama_errors_total\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE",
                      "crossSeriesReducer": "REDUCE_SUM"
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "yPos": 3,
        "width": 6,
        "height": 3,
        "widget": {
          "title": "API Latency (ms) - p99",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"ollama_request_duration_seconds\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_PERCENTILE_99",
                      "crossSeriesReducer": "REDUCE_MEAN"
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "xPos": 6,
        "yPos": 3,
        "width": 6,
        "height": 3,
        "widget": {
          "title": "Instance Count",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"run.googleapis.com/instance_count\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MAX"
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "yPos": 6,
        "width": 6,
        "height": 3,
        "widget": {
          "title": "Database CPU (%)",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"cloudsql.googleapis.com/database/cpu/utilization\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "xPos": 6,
        "yPos": 6,
        "width": 6,
        "height": 3,
        "widget": {
          "title": "Database Connections",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"ollama_db_active_connections\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MAX"
                    }
                  }
                }
              }
            ]
          }
        }
      }
    ]
  }
}
```

### Inference Performance Dashboard

```json
{
  "displayName": "Ollama Production - Inference Performance",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "width": 6,
        "height": 3,
        "widget": {
          "title": "Inference Requests by Model",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"ollama_inference_requests_total\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE",
                      "groupByFields": ["metric.model"]
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "xPos": 6,
        "width": 6,
        "height": 3,
        "widget": {
          "title": "Inference Latency by Model (p99)",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"ollama_inference_duration_seconds\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_PERCENTILE_99",
                      "groupByFields": ["metric.model"]
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "yPos": 3,
        "width": 12,
        "height": 3,
        "widget": {
          "title": "Tokens Generated per Model",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"ollama_tokens_generated_total\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE",
                      "groupByFields": ["metric.model"]
                    }
                  }
                }
              }
            ]
          }
        }
      }
    ]
  }
}
```

### Cache Performance Dashboard

```json
{
  "displayName": "Ollama Production - Cache Performance",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "width": 6,
        "height": 3,
        "widget": {
          "title": "Cache Hit Rate (%)",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"ollama_cache_hit_ratio\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN",
                      "groupByFields": ["metric.cache_type"]
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "xPos": 6,
        "width": 6,
        "height": 3,
        "widget": {
          "title": "Redis Memory Usage",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"redis.googleapis.com/memory/usage\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MAX"
                    }
                  }
                }
              }
            ]
          }
        }
      }
    ]
  }
}
```

---

## SLOs & SLIs

### Service Level Objectives

```yaml
slos:
  # Availability SLO
  - name: "API Availability"
    target: 99.9%  # 43 minutes downtime/month
    description: "API responds successfully to requests"
    sli: "success_rate"
    window: "30d"

  # Latency SLO
  - name: "API Latency"
    target: 99%  # 99% of requests < 500ms
    description: "API responds within acceptable time"
    sli: "latency_p99 < 500ms"
    window: "30d"

  # Error Rate SLO
  - name: "Error Rate"
    target: 99.9%  # < 0.1% error rate
    description: "Requests complete without error"
    sli: "error_rate < 0.001"
    window: "30d"

  # Model Availability SLO
  - name: "Model Availability"
    target: 99.5%  # Models available 99.5% of time
    description: "Inference models are ready"
    sli: "model_ready_ratio"
    window: "7d"
```

### Service Level Indicators

```python
# SLI Calculation

# 1. Success Rate (requests returning 2xx)
success_rate = (
    successful_requests / total_requests
)
# Alert if < 99.9%

# 2. Latency (p99 response time)
latency_p99 = percentile(response_times, 99)
# Alert if > 500ms

# 3. Error Rate
error_rate = (
    failed_requests / total_requests
)
# Alert if > 0.1%

# 4. Model Ready Ratio
model_ready_ratio = (
    ready_models / total_models
)
# Alert if < 99%

# 5. Database Connection Pool
pool_exhaustion = (
    active_connections / max_connections
)
# Alert if > 90%
```

### Error Budget

```
Total Budget: 100% - 99.9% = 0.1% (43 minutes/month)

Usage:
- Planned maintenance: 30 minutes
- Incident buffer: 13 minutes

Status: On track
Remaining: 0 minutes (use next month for buffer)
```

---

## Data Retention

### Metrics Retention Policy

```yaml
retention:
  # Standard resolution (60s)
  raw_metrics:
    retention: 24h  # 1 day
    storage: cloud_monitoring_disk

  # 5-minute resolution
  coarse_metrics:
    retention: 30d  # 1 month
    storage: cloud_monitoring

  # 1-hour resolution
  archived_metrics:
    retention: 1y  # 1 year
    storage: cloud_storage_archive

  # Critical alerts (always keep)
  incident_data:
    retention: 2y  # 2 years
    storage: firestore

  # Logs
  application_logs:
    retention: 30d  # 1 month
    storage: cloud_logging

  # Security logs
  security_logs:
    retention: 2y  # 2 years
    storage: cloud_storage_compliance

  # Audit logs
  audit_logs:
    retention: indefinite  # Forever (compliance)
    storage: cloud_storage_compliance
```

### Log Aggregation

```bash
# Cloud Logging filters

# All errors
resource.type="cloud_run_revision" AND severity="ERROR"

# Specific service
resource.labels.service_name="ollama" AND resource.labels.revision_name="ollama-*"

# Performance logs
resource.type="cloud_run_revision" AND jsonPayload.duration_ms > 5000

# Security events
protoPayload.methodName="storage.objects.create" OR protoPayload.methodName="storage.objects.delete"

# Database operations
resource.type="cloudsql_database" AND (protoPayload.status.code != 0 OR query_duration_ms > 5000)
```

---

## Document Updates

- **Last Updated**: January 13, 2026
- **Next Review**: January 20, 2026
- **Owner**: Platform Engineering
- **Version**: 1.0

**For Questions**: Contact the SRE team or open an issue in the monitoring repository.
