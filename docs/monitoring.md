# Monitoring & Observability Configuration

## Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ollama-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

## Key Metrics

### Inference Performance
- `ollama_inference_duration_seconds`: Inference latency distribution
- `ollama_tokens_per_second`: Throughput metric
- `ollama_cache_hit_ratio`: Cache efficiency

### Resource Usage
- `ollama_gpu_memory_used_bytes`: GPU memory consumption
- `ollama_gpu_utilization_percent`: GPU utilization
- `ollama_system_memory_used_bytes`: System memory usage

### Request Patterns
- `ollama_requests_total`: Total requests processed
- `ollama_request_errors_total`: Failed requests
- `ollama_request_queue_depth`: Pending requests

## Grafana Dashboards

Pre-built dashboards for:
- System resources (CPU, RAM, GPU)
- Model performance metrics
- Request patterns and error rates
- Cost analysis (compute time, energy)

## Alert Rules

```yaml
# Critical alerts
- inference_latency_high: p99 > 1000ms
- gpu_memory_exhausted: usage > 95%
- error_rate_high: errors > 5%
- service_unhealthy: health check failures
```

## Log Aggregation (Loki)

Structured logging with labels:
- `job`: Service identifier
- `level`: Log level (ERROR, WARN, INFO, DEBUG)
- `model`: Model name
- `trace_id`: Request trace ID
