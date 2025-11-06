# WSL Dev Quickstart

Develop on Windows using **WSL (Ubuntu)** with a dev binary isolated from the system service.

## Start the server inside WSL

Your build may or may not support the `--host` flag.
- If it **does not**, it will listen on **127.0.0.1:11434** by default.
- If it **does**, you can run on **127.0.0.1:11436** to keep it separate from the system service.

```bash
# inside WSL
cd ~/code/ollama

# separate directory for development models
export OLLAMA_MODELS="$HOME/.ollama-dev/models"
mkdir -p "$OLLAMA_MODELS"

# builds WITHOUT --host (most common on older binaries):
./bin/ollama serve
# → listens on 127.0.0.1:11434

# builds WITH --host (newer binaries):
# ./bin/ollama serve --host 127.0.0.1:11436
