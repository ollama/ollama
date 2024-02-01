#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit

PORT="${PORT:-8080}"
if test -f docker_secret_key && test "$WEBUI_SECRET_KEY" = ""; then
  echo Using generated DOCKER_SECRET_KEY
  WEBUI_SECRET_KEY=`cat docker_secret_key`
fi

WEBUI_SECRET_KEY="$WEBUI_SECRET_KEY" exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --forwarded-allow-ips '*'
