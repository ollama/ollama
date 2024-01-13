#!/usr/bin/env bash

PORT="${PORT:-8080}"
uvicorn main:app --host 0.0.0.0 --port $PORT --forwarded-allow-ips '*'
