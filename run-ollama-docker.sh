#!/bin/bash

host_port=11434
container_port=11434

read -r -p "Do you want ollama in Docker with GPU support? (y/n): " use_gpu

docker rm -f ollama || true
docker pull ollama/ollama:latest

docker_args="-d -v ollama:/root/.ollama -p $host_port:$container_port --name ollama ollama/ollama"

if [ "$use_gpu" = "y" ]; then
    docker_args="--gpus=all $docker_args"
fi

docker run $docker_args

docker image prune -f