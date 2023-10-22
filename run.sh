docker stop ollama-webui || true
docker rm ollama-webui || true
docker build -t ollama-webui .
docker run -d -p 3000:8080 --name ollama-webui --restart always ollama-webui
docker image prune -f