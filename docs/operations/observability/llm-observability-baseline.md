# LLM Observability Baseline

Issue: #243

This baseline establishes production observability for:
- End-to-end request latency
- Token usage (prompt, completion, total)
- Cache/retrieval hit signals
- Failure reason classification
- Per-request trace IDs

## Metrics Endpoint

The server now exposes Prometheus text exposition at:

- `GET /metrics`

Trace correlation header returned on all requests:

- `X-Trace-Id`

## Core Metrics

- `ollama_http_requests_total{route,status}`
- `ollama_http_failures_total{route,reason}`
- `ollama_http_request_duration_ms_bucket{route,le}`
- `ollama_tokens_total{route,kind}`
- `ollama_signal_hits_total{route,signal}`
- `ollama_http_inflight_requests`

## Dashboard Signals

Recommended baseline panels:
- Request volume by route/status
- P50/P95/P99 request latency by route
- Failure rate by route and reason
- Prompt/completion token burn over time
- Cache/retrieval hit ratio
- In-flight request pressure

## Alert Thresholds

Suggested initial thresholds:
- P95 latency > 2500ms for 10m
- 5xx error ratio > 2% for 5m
- Zero cache/retrieval hit over 30m where previously non-zero
- Token burn spike > 3x baseline over 15m

Alert examples are provided in:

- `docs/operations/observability/llm-observability-alerts.yml`

## Incident Drill

Runbook and drill procedure:

- `docs/operations/observability/llm-observability-incident-drill.md`

## Notes

- Token metrics are extracted from response payload fields (`prompt_eval_count`, `eval_count`, and OpenAI-style token fields when present).
- Streaming responses are parsed from final NDJSON chunk when possible.
- Cache/retrieval hit signals are sourced from response headers (`X-Cache-Hit`, `X-Retrieval-Hit`).
