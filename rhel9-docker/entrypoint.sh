#!/bin/bash

ollama serve & \
sleep 2 && \
ollama pull $MODEL && \
litellm --model ollama/$MODEL --config ./config.yaml --host $HOST --port $PORT 