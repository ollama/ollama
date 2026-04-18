#!/bin/bash
# ============================================================================
# Setup secrets for Ollama deployment
# ============================================================================
set -e

mkdir -p secrets
chmod 700 secrets

# Helper to create secret if missing
create_secret() {
  local file=$1
  local desc=$2
  if [ -f "$file" ]; then
    echo "[SKIP] $desc already exists ($file)"
  else
    openssl rand -base64 32 > "$file"
    chmod 600 "$file"
    echo "[OK] Created $desc ($file)"
  fi
}

create_secret secrets/db_password.txt "Postgres password"
create_secret secrets/redis_password.txt "Redis password"
create_secret secrets/grafana_password.txt "Grafana admin password"
create_secret secrets/grafana_db_password.txt "Grafana DB password"

echo "[INFO] Place your GCP service account key at secrets/gcp-service-account.json"
echo "[INFO] Ensure correct permissions: chmod 600 secrets/gcp-service-account.json"

echo "[DONE] Secrets setup complete."
