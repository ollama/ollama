# Docker

It's possible to run Ollama with Docker or Docker Compose.

The official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama)
`ollama/ollama` is available on Docker Hub.

## Docker Hub

To interact with the official Ollama images, see the below.

### CPU

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### GPU

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## Docker Compose

There is an example `docker-compose.yaml` file in `examples/docker-compose`.
Here's how to use it:

```bash
git clone git@github.com:jmorganca/ollama.git
cd ollama/examples/docker-compose
docker compose up --wait --detach
```
