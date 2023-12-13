#!/bin/bash
# Run the ollama service and wait 2 seconds for it to start
ollama serve & \
sleep 2 && \
# Pull the model declared in the MODEL environment variable
ollama pull $MODEL && \
# Run litellm with the environment variables
litellm --model ollama/$MODEL --config ./config.yaml --host $HOST --port $PORT 