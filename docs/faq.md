# FAQ

## How can I use Ollama in Visual Studio Code?

There is already a large collection of plugins available for VSCode as well as other editors that leverage Ollama. You can see the list of [extensions & plugins](https://github.com/jmorganca/ollama#extensions--plugins) at the bottom of the main repository readme.

## How do I use Ollama with GPU acceleration in Docker?

The Ollama Docker container can be configured with GPU acceleration in Linux or Windows (with WSL2). This requires the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). See [ollama/ollama](https://hub.docker.com/r/ollama/ollama) for more details.

GPU acceleration is not available for Docker Desktop in macOS due to the lack of GPU passthrough and emulation.
