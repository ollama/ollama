#!/bin/bash

read -r -p "Do you want ollama in Docker with GPU support? (y/n): " use_gpu

docker rm -f ollama || true
docker pull ollama/ollama:latest

docker_args="-d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama"

if [ "$use_gpu" == "y" ]; then
    docker_args+=" --gpus=all"
fi

docker run "$docker_args"

docker image prune -f
