# LLM Observability Incident Drill

Issue: #243

## Goal

Validate that the observability baseline detects and routes an incident using:
- Trace IDs
- Metrics and alerts
- Dashboard triage
- Runbook execution

## Drill Scenario

Inject a controlled latency and error spike on inference endpoints.

## Preconditions

- Ollama server running with `/metrics` exposed
- Prometheus scraping `/metrics`
- Alerts loaded from `llm-observability-alerts.yml`
- Dashboard imported and visible

## Steps

1. Baseline capture (10 minutes):
   - Confirm normal request rate and error ratio.
   - Record baseline p95 latency and token throughput.

2. Inject synthetic load:
   - Generate sustained inference requests to `/api/chat` and `/v1/chat/completions`.
   - Introduce fault (timeout, dependency failure, or controlled error path).

3. Observe alert behavior:
   - Verify `OllamaHighP95Latency` and/or `OllamaHighServerErrorRate` fire.
   - Verify annotations include route/severity context.

4. Trace correlation:
   - Capture `X-Trace-Id` from failing requests.
   - Correlate trace IDs with server logs for rapid RCA.

5. Recovery:
   - Remove fault condition.
   - Verify alert auto-resolve and return-to-baseline metrics.

6. Post-drill review:
   - Time to detect (TTD)
   - Time to identify (TTI)
   - Time to mitigate (TTM)
   - Missing telemetry gaps and follow-up actions

## Exit Criteria

- Alerts fire and resolve as expected.
- At least one incident path is correlated using `X-Trace-Id`.
- RCA can be completed from dashboard + logs within target SLO.
- Follow-up tasks are documented.
