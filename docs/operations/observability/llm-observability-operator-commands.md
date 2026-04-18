# LLM Observability Operator Commands

Issue: #243

These commands provide a repeatable operator workflow for validating observability baseline in production-like environments.

## Variables

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

## A) Quick Health and Trace Check

```bash
curl -i -sS "${OLLAMA_BASE_URL}/api/version"
```

Check response headers include:
- `X-Trace-Id`

## B) Metrics Family Presence

```bash
curl -sS "${OLLAMA_BASE_URL}/metrics" | egrep '^ollama_http_requests_total|^ollama_http_failures_total|^ollama_http_request_duration_ms_bucket|^ollama_tokens_total|^ollama_signal_hits_total|^ollama_http_inflight_requests'
```

## C) Controlled Failure Injection (Safe)

```bash
for i in $(seq 1 50); do
  curl -sS -X POST "${OLLAMA_BASE_URL}/api/chat" \
    -H 'Content-Type: application/json' \
    -d '{"model":"nonexistent-model-for-drill","messages":[{"role":"user","content":"hello"}],"stream":false}' >/dev/null || true
done
```

This should increment request/failure counters and exercise alert conditions.

## D) Token Signal Generation

```bash
# Replace with a known local model in your environment.
curl -sS -X POST "${OLLAMA_BASE_URL}/api/chat" \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Provide a short summary."}],"stream":false}' | jq .
```

Then check token counters:

```bash
curl -sS "${OLLAMA_BASE_URL}/metrics" | grep '^ollama_tokens_total'
```

## E) Snapshot Export for Evidence

```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p /tmp/ollama-observability-evidence
curl -sS "${OLLAMA_BASE_URL}/metrics" > "/tmp/ollama-observability-evidence/metrics-${ts}.prom"
curl -i -sS "${OLLAMA_BASE_URL}/api/version" > "/tmp/ollama-observability-evidence/version-${ts}.txt"
ls -lah /tmp/ollama-observability-evidence
```

Attach the files to issue #243 as deployment validation evidence.
