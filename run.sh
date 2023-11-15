docker stop ollama-webui || true
docker rm ollama-webui || true
docker build -t ollama-webui .
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway --name ollama-webui --restart always ollama-webui
docker image prune -f