#!/bin/bash

image_name="ollama-webui"
container_name="ollama-webui"

docker build -t "$image_name" .
docker stop "$container_name" &>/dev/null || true
docker rm "$container_name" &>/dev/null || true

docker run -d -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -v "${image_name}:/app/backend/data" \
    --name "$container_name" \
    --restart always \
    "$image_name"

docker image prune -f
