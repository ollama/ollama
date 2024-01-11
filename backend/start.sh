#!/usr/bin/env bash

SERVER_PORT="${SERVER_PORT:-8080}"
uvicorn main:app --host 0.0.0.0 --port $SERVER_PORT --forwarded-allow-ips '*'
