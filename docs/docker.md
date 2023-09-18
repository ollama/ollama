# Docker Commands

## Simple commands
```
docker build -t ollama-image -f Dockerfile.cuda .
```

```
docker run --name ollama-container --gpus all -v /home/user/datadir/ollama:/home/ollama -p 11434:11434 ollama-image
```