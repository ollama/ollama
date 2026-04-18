#!/usr/bin/env bash

set -euo pipefail

detect_repo_slug() {
  if [[ -n "${GITHUB_REPOSITORY:-}" ]]; then
    printf '%s' "$GITHUB_REPOSITORY"
    return 0
  fi

  local remote_url=""
  if ! remote_url="$(git remote get-url origin 2>/dev/null)"; then
    return 1
  fi

  remote_url="${remote_url%.git}"
  if [[ "$remote_url" == *"github.com/"* ]]; then
    printf '%s' "${remote_url#*github.com/}"
    return 0
  fi

  if [[ "$remote_url" == *"github.com:"* ]]; then
    printf '%s' "${remote_url#*github.com:}"
    return 0
  fi

  return 1
}

resolve_token_from_file() {
  local token_file="${1:-}"
  local token=""

  if [[ -z "$token_file" || ! -f "$token_file" ]]; then
    return 1
  fi

  token="$(tr -d '\r' < "$token_file" | sed -e 's/[[:space:]]*$//')"
  [[ -n "$token" ]] || return 1
  printf '%s' "$token"
}

resolve_token_from_git_credentials() {
  local repo_slug="${1:-}"
  local request=$'protocol=https\nhost=github.com\n'

  if [[ -n "$repo_slug" ]]; then
    request+="path=${repo_slug}.git"$'\n'
  fi
  request+=$'\n'

  local output=""
  if ! output="$(git credential fill <<< "$request" 2>/dev/null)"; then
    return 1
  fi

  local line token=""
  while IFS= read -r line; do
    if [[ "$line" == password=* ]]; then
      token="${line#password=}"
      break
    fi
  done <<< "$output"

  [[ -n "$token" ]] || return 1
  printf '%s' "${token%$'\r'}"
}

resolve_token_from_gsm() {
  if [[ ! "${OLLAMA_GSM_ENABLED:-}" =~ ^(true|1|yes)$ ]]; then
    return 1
  fi

  if ! command -v gcloud >/dev/null 2>&1; then
    return 1
  fi

  local project_id="${GSM_PROJECT:-${OLLAMA_GSM_PROJECT_ID:-gcp-eiq}}"
  local secret_name="${GSM_SECRET_NAME:-${OLLAMA_GSM_SECRET_NAME:-prod-github-token}}"
  local token=""

  if ! token="$(gcloud secrets versions access latest --secret="$secret_name" --project="$project_id" 2>/dev/null)"; then
    return 1
  fi

  token="$(printf '%s' "$token" | tr -d '\r' | sed -e 's/[[:space:]]*$//')"
  [[ -n "$token" ]] || return 1
  printf '%s' "$token"
}

resolve_github_token() {
  local repo_slug="${1:-}"
  local token_file="${OLLAMA_GITHUB_TOKEN_FILE:-${GITHUB_TOKEN_FILE:-}}"
  local token=""

  if token="$(resolve_token_from_file "$token_file" 2>/dev/null)"; then
    printf '%s' "$token"
    return 0
  fi

  if token="$(resolve_token_from_git_credentials "$repo_slug" 2>/dev/null)"; then
    printf '%s' "$token"
    return 0
  fi

  if token="$(resolve_token_from_gsm 2>/dev/null)"; then
    printf '%s' "$token"
    return 0
  fi

  if [[ -n "${OLLAMA_GITHUB_TOKEN:-}" ]]; then
    printf '%s' "${OLLAMA_GITHUB_TOKEN%$'\r'}"
    return 0
  fi

  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    printf '%s' "${GITHUB_TOKEN%$'\r'}"
    return 0
  fi

  if [[ -n "${GH_TOKEN:-}" ]]; then
    printf '%s' "${GH_TOKEN%$'\r'}"
    return 0
  fi

  return 1
}

repo_slug=""
if repo_slug="$(detect_repo_slug 2>/dev/null)"; then
  :
fi

if token="$(resolve_github_token "$repo_slug")"; then
  printf '%s' "$token"
else
  echo "ERROR: unable to resolve a GitHub token from token file, git credentials, GSM, or environment" >&2
  exit 1
fi
