docker rm -f ollama || true
docker pull ollama/ollama
# CPU Only
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
# GPU Support
# docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker image prune -f