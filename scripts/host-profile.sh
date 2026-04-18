#!/bin/bash

# Host profile loader for target-server-local execution.

load_host_profile() {
  local project_root="${1:-$(pwd)}"
  local profile_file="${HOST_PROFILE_FILE:-}"

  if [ -z "$profile_file" ] && [ -n "${TARGET_ENV:-}" ]; then
    profile_file="${project_root}/config/hosts/${TARGET_ENV}.env"
  fi

  if [ -z "$profile_file" ] || [ ! -f "$profile_file" ]; then
    return 0
  fi

  set -a
  # shellcheck disable=SC1090
  source "$profile_file"
  set +a

  if [ -z "${OLLAMA_HOST:-}" ] && [ -n "${BACKEND_HOST:-}" ]; then
    export OLLAMA_HOST="http://${BACKEND_HOST}:11434"
  fi

  export HOST_PROFILE_FILE="$profile_file"
}
