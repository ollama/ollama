#!/usr/bin/env bash
set -euo pipefail

# Validate baseline observability acceptance signals for issue #243.
# This script is safe to run repeatedly and emits timestamped evidence artifacts.

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
EVIDENCE_DIR="${EVIDENCE_DIR:-/tmp/ollama-observability-evidence}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${EVIDENCE_DIR}"

version_file="${EVIDENCE_DIR}/version-${TIMESTAMP}.txt"
metrics_file="${EVIDENCE_DIR}/metrics-${TIMESTAMP}.prom"
summary_file="${EVIDENCE_DIR}/summary-${TIMESTAMP}.txt"

required_metrics=(
  "ollama_http_requests_total"
  "ollama_http_failures_total"
  "ollama_http_request_duration_ms_bucket"
  "ollama_tokens_total"
  "ollama_signal_hits_total"
  "ollama_http_inflight_requests"
)

echo "[info] OLLAMA_BASE_URL=${OLLAMA_BASE_URL}"
echo "[info] EVIDENCE_DIR=${EVIDENCE_DIR}"

# Capture version response headers and body for trace-id verification evidence.
curl -i -sS "${OLLAMA_BASE_URL}/api/version" >"${version_file}"

# Capture current metrics snapshot.
curl -sS "${OLLAMA_BASE_URL}/metrics" >"${metrics_file}"

trace_ok=0
if grep -qi '^X-Trace-Id:' "${version_file}"; then
  trace_ok=1
fi

missing=0
{
  echo "# Observability Baseline Validation Summary"
  echo "timestamp=${TIMESTAMP}"
  echo "base_url=${OLLAMA_BASE_URL}"
  echo "version_file=${version_file}"
  echo "metrics_file=${metrics_file}"
  echo ""
  echo "trace_header=$([[ ${trace_ok} -eq 1 ]] && echo PASS || echo FAIL)"
  echo ""
  echo "metrics_families:"

  for metric in "${required_metrics[@]}"; do
    if grep -q "^# HELP ${metric}\b" "${metrics_file}" || grep -q "^${metric}\b" "${metrics_file}"; then
      echo "  - ${metric}: PASS"
    else
      echo "  - ${metric}: FAIL"
      missing=1
    fi
  done
} >"${summary_file}"

cat "${summary_file}"

if [[ ${trace_ok} -ne 1 || ${missing} -ne 0 ]]; then
  echo "[error] Observability baseline validation failed. See ${summary_file}" >&2
  exit 1
fi

echo "[ok] Observability baseline validation passed."
echo "[ok] Attach ${summary_file}, ${version_file}, and ${metrics_file} to issue #243 evidence comment."
