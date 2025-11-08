#!/bin/bash

# Model Configuration for Agent System
# Edit this file to customize which models are automatically pulled on startup

# Required models - these will be automatically pulled if not present
# Format: "model:tag" or "model" (defaults to :latest)

REQUIRED_MODELS=(
    # Primary model for general tasks (2.0 GB)
    "llama3.2:latest"

    # Fast coding model (986 MB)
    "qwen2.5:1.5b"

    # Reasoning/research model (1.1 GB)
    "deepseek-r1:1.5b"

    # Analysis model (1.6 GB)
    "gemma2:2b"

    # Technical tasks model (2.2 GB)
    "phi3:mini"

    # Smallest/fastest model for quick responses (397 MB)
    "qwen2.5:0.5b"
)

# Optional: Add more models here
# Examples:
# "mistral:latest"
# "codellama:latest"
# "llama3.2:3b"

# Total size: ~8.2 GB for all models above
