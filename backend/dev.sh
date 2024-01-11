SERVER_PORT="${SERVER_PORT:-8080}"
uvicorn main:app --port $SERVER_PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload