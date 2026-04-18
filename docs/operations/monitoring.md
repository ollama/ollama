# Monitoring & Alerting

Set up observability for Ollama.

## Metrics

Key metrics to monitor:

### API Metrics

```
ollama_api_requests_total{method,endpoint,status}
ollama_api_request_duration_seconds{method,endpoint,quantile}
ollama_api_errors_total{error_type}
```

### Inference Metrics

```
ollama_inference_requests_total{model,status}
ollama_inference_duration_seconds{model,quantile}
ollama_tokens_generated_total{model}
ollama_inference_errors_total{model,error_type}
```

### System Metrics

```
ollama_model_cache_hit_ratio
ollama_database_connections{pool_name,status}
ollama_redis_operations_total{operation,status}
ollama_health_check_status{region}
```

## Health Checks

Configure health checks:

=== "Primary Region"
`bash
    gcloud compute health-checks create http ollama-health-check \
      --port=8000 \
      --request-path=/api/v1/health \
      --check-interval=10s \
      --timeout=5s \
      --healthy-threshold=2 \
      --unhealthy-threshold=3
    `

=== "Load Balancer"
`bash
    gcloud compute backend-services update prod-ollama-api-backend \
      --health-checks=ollama-health-check \
      --global
    `

## Dashboards

Create Grafana dashboards:

1. **API Performance**: Request latency, error rate, throughput
2. **Inference**: Model performance, token generation rate, queue depth
3. **Infrastructure**: CPU, memory, disk, network usage
4. **Database**: Query latency, connection pool, transaction rate
5. **Failover**: Health status by region, failover events, latency by region

Dashboard JSON available in `monitoring/dashboards/`.

## Alerting

Configure alerts for critical conditions:

| Alert              | Condition          | Action             |
| ------------------ | ------------------ | ------------------ |
| High Error Rate    | 5xx errors > 1%    | Page on-call       |
| Slow API           | p95 latency > 1s   | Notify Slack       |
| Model Queue        | Queue depth > 100  | Scale up           |
| Low Cache Hit      | Hit ratio < 70%    | Investigate        |
| Region Unavailable | Health checks fail | Failover triggered |
| High Memory        | Usage > 85%        | Alert, scale up    |

Example alert rule (Prometheus):

```yaml
groups:
  - name: ollama_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(ollama_api_errors_total[5m]) > 0.01
        for: 5m
        annotations:
          summary: "High API error rate"
```

## Logging

Structured logging with context:

```python
import structlog

log = structlog.get_logger()

log.info(
    "inference_completed",
    model="llama2",
    tokens=150,
    latency_ms=2100,
    request_id="req_abc123",
    user_id="user_123",
)
```

See [Operations Runbook](runbooks.md) for debugging procedures.
