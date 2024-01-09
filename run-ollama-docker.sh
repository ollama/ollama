#!/bin/bash

read -r -p "Do you want ollama in docker with GPU support? (y/n): " use_gpu

docker rm -f ollama || true
docker pull ollama/ollama:latest

if [ "$use_gpu" == "y" ]; then
    docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
else
    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
fi

docker image prune -f
