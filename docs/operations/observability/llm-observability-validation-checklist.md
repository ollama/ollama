# LLM Observability Deployment Validation Checklist

Issue: #243

Use this checklist to complete final environment-level acceptance and close the issue.

## Scope

This validates:
- dashboard live
- alert rules tested
- incident drill completed

## Prerequisites

- Ollama server deployed with observability changes.
- Prometheus scraping `GET /metrics` from Ollama.
- Alert rules loaded from `llm-observability-alerts.yml`.
- Grafana connected to Prometheus datasource.

## 1) Metrics and Trace-ID Smoke Validation

Quick path (single command):

```bash
./scripts/validate_observability_baseline.sh
```

This writes evidence files under `/tmp/ollama-observability-evidence` and exits non-zero when required checks fail.

Manual path:

Run from an operator shell:

```bash
OLLAMA_BASE_URL="http://127.0.0.1:11434"

# 1. Trace header exists
curl -i -sS "${OLLAMA_BASE_URL}/api/version" | grep -i 'X-Trace-Id:'

# 2. Metrics endpoint responds
curl -sS "${OLLAMA_BASE_URL}/metrics" | grep -E '^# HELP ollama_http_requests_total|^ollama_http_requests_total'

# 3. Core observability metric families are present
for metric in \
  ollama_http_requests_total \
  ollama_http_failures_total \
  ollama_http_request_duration_ms_bucket \
  ollama_tokens_total \
  ollama_signal_hits_total \
  ollama_http_inflight_requests
  do
    curl -sS "${OLLAMA_BASE_URL}/metrics" | grep -q "^${metric}" && echo "OK ${metric}" || echo "MISSING ${metric}"
  done
```

Expected result:
- `X-Trace-Id` is present on `/api/version` response.
- All listed metric families are present on `/metrics`.

## 2) Grafana Dashboard Live Validation

Import dashboard JSON:
- `docs/operations/observability/llm-observability-dashboard.json`

Validate the following panels render data:
- HTTP Requests by Route/Status
- P95 Latency (ms)
- Failure Rate
- Token Throughput
- Cache/Retrieval Hits
- In-flight Requests

Evidence:
- Screenshot or exported panel JSON with timestamp.

## 3) Alert Rule Validation

Load rule file:
- `docs/operations/observability/llm-observability-alerts.yml`

Validation approach:
- Confirm rules are loaded in Prometheus and visible in Alertmanager.
- Trigger at least one warning and one critical path.

Suggested drills:

```bash
# Example load loop producing controlled request pressure and error responses.
OLLAMA_BASE_URL="http://127.0.0.1:11434"
for i in $(seq 1 200); do
  curl -sS -X POST "${OLLAMA_BASE_URL}/api/chat" \
    -H 'Content-Type: application/json' \
    -d '{"model":"nonexistent-model-for-drill","messages":[{"role":"user","content":"drill"}],"stream":false}' >/dev/null || true
done
```

Expected result:
- `OllamaHighServerErrorRate` or latency-related alert fires as expected.
- Alert resolves after load/fault condition is removed.

## 4) Incident Drill Completion

Execute runbook:
- `docs/operations/observability/llm-observability-incident-drill.md`

Capture:
- Time to detect (TTD)
- Time to identify (TTI)
- Time to mitigate (TTM)
- Correlated `X-Trace-Id` from a failing request to logs
- Follow-up remediation tasks

## 5) Closure Gate

Mark issue #243 complete only when all are true:
- [ ] Dashboard is live and validated
- [ ] Alert rules loaded and fired in test
- [ ] Incident drill completed with evidence
- [ ] Follow-up actions documented

## Evidence Template

Use this in the issue comment:

```markdown
## #243 Environment Validation Evidence

- Metrics smoke: PASS/FAIL
- Dashboard import: PASS/FAIL
- Alerts test: PASS/FAIL
- Incident drill: PASS/FAIL

### Metrics smoke
- Timestamp:
- Endpoint:
- Trace header observed:

### Dashboard
- Grafana link:
- Screenshot link:

### Alerts
- Fired alerts:
- Resolve confirmation:

### Drill
- TTD:
- TTI:
- TTM:
- Trace-ID correlation evidence:

### Follow-ups
- [ ] item 1
- [ ] item 2
```
