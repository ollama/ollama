#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-.github/issue-orchestrator.iac.json}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config file not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p .github/state
LOCK_FILE="$(python3 - "$CONFIG_PATH" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = json.load(f)
print(cfg.get('governance', {}).get('idempotent_lock_file', '.github/state/.issue_orchestrator.lock'))
PY
)"

mkdir -p "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "INFO: issue orchestrator already running, skipping (idempotent lock active)."
  exit 0
fi

readarray -t CFG < <(python3 - "$CONFIG_PATH" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = json.load(f)
repo = cfg.get('repository', {})
exe = cfg.get('execution', {})
auth = cfg.get('auth', {})
print(repo.get('primary', 'kushin77/ollama'))
print(repo.get('fallback', 'ollama/ollama'))
print('true' if repo.get('allow_fallback', False) else 'false')
print('true' if exe.get('all_severities', True) else 'false')
print(str(int(exe.get('max_issues', 100))))
print('true' if exe.get('execute', True) else 'false')
print(','.join(exe.get('auto_close_labels', ['duplicate', 'invalid', 'wontfix'])))
print(exe.get('output', '.github/orchestrator_report_autonomous.json'))
print('true' if auth.get('ollama_gsm_enabled', True) else 'false')
print(auth.get('gsm_project', 'gcp-eiq'))
print(auth.get('gsm_secret_name', 'prod-github-token'))
PY
)

PRIMARY_REPO="${CFG[0]}"
FALLBACK_REPO="${CFG[1]}"
ALLOW_FALLBACK="${CFG[2]}"
ALL_SEVERITIES="${CFG[3]}"
MAX_ISSUES="${CFG[4]}"
DO_EXECUTE="${CFG[5]}"
AUTO_CLOSE_LABELS="${CFG[6]}"
OUTPUT_PATH="${CFG[7]}"
GSM_ENABLED="${CFG[8]}"
GSM_PROJECT_VAL="${CFG[9]}"
GSM_SECRET_VAL="${CFG[10]}"

export OLLAMA_GSM_ENABLED="$GSM_ENABLED"
export GSM_PROJECT="$GSM_PROJECT_VAL"
export GSM_SECRET_NAME="$GSM_SECRET_VAL"

ARGS=(
  "cmd/github-issues/orchestrator_enhanced.py"
  "--repo" "$PRIMARY_REPO"
  "--fallback" "$FALLBACK_REPO"
  "--max-issues" "$MAX_ISSUES"
  "--auto-close-labels" "$AUTO_CLOSE_LABELS"
  "--output" "$OUTPUT_PATH"
)

if [[ "$ALLOW_FALLBACK" == "true" ]]; then
  ARGS+=("--allow-fallback")
fi
if [[ "$ALL_SEVERITIES" == "true" ]]; then
  ARGS+=("--all-severities")
else
  ARGS+=("--severity" "high")
fi
if [[ "$DO_EXECUTE" == "true" ]]; then
  ARGS+=("--execute")
fi

echo "INFO: running autonomous issue orchestrator against $PRIMARY_REPO"
python3 "${ARGS[@]}"

echo "INFO: orchestration complete. report=$OUTPUT_PATH"
