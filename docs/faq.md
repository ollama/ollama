# FAQ

## How can I view the logs?

Review the [Troubleshooting](./troubleshooting.md) docs for more about using logs.

## How can I expose Ollama on my network?

To expose Ollama to another host, you need to use the `OLLAMA_HOST` environment variable. See [Serving Ollama in the CLI Documentation](./cli.md#serving-ollama) for more on using this.

## How can I allow additional web origins to access Ollama?

You will need to use `OLLAMA_ORIGINS` to allow additional web origins. See [Serving Ollama in the CLI Documentation](./cli.md#serving-ollama) for more on using this.

## Where are models stored?

- macOS: All model data is stored under `~/.ollama/models`.
- Linux: All model data is stored under `/usr/share/ollama/.ollama/models`

See [the CLI Documentation](./cli.md) for more on this.

## Does Ollama send my prompts and answers back to Ollama.ai to use in any way?

No, Ollama runs 100% locally wherever you have installed it. Other than downloading the models, we never reach out anywhere, and models are never updated using your questions and answers.

## How can I use Ollama in Visual Studio Code?

There is already a large collection of plugins available for VSCode as well as other editors that leverage Ollama. You can see the list of [extensions & plugins](https://github.com/jmorganca/ollama#extensions--plugins) at the bottom of the main repository readme.

## How do I use Ollama behind a proxy?

Ollama is compatible with proxy servers if `HTTP_PROXY` or `HTTPS_PROXY` are configured. See [Serving Ollama in the CLI Documentation](./cli.md#serving-ollama) for more on using this.

### How do I use Ollama behind a proxy in Docker?

The Ollama Docker container image can be configured to use a proxy by passing `-e HTTPS_PROXY=https://proxy.example.com` when starting the container. See [Serving Ollama in the CLI Documentation](./cli.md#serving-ollama) for more on using this.

## How do I use Ollama with GPU acceleration in Docker?

The Ollama Docker container can be configured with GPU acceleration in Linux or Windows (with WSL2). This requires the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). See [ollama/ollama](https://hub.docker.com/r/ollama/ollama) for more details.

GPU acceleration is not available for Docker Desktop in macOS due to the lack of GPU passthrough and emulation.
