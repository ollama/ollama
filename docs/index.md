# Ollama

## Run Large Language Models Locally

Ollama lets you run powerful large language models (LLMs) on your own hardware. Get started with AI without sending your data to third-party services.

## Key Features

- **Run models locally** on your Mac, [Windows](./installation/windows.md), or [Linux](./installation/linux.md) machine
- **No cloud required** - your data stays on your device
- **Easy to use** - [simple commands](./getting_started/quickstart.md) to download and run models
- **Customizable** - [create](./getting_started/modelfile.md) and [modify](./getting_started/examples.md) your own models
- **API access** - [integrate](api.md) with your applications

## Quick Start

### 1. Install Ollama

Choose your platform:

- **macOS**: [Download](https://ollama.com/download/Ollama-darwin.zip)
- **Windows**: [Download](https://ollama.com/download/OllamaSetup.exe)
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **Docker**: `docker pull ollama/ollama`

### 2. Run a Model

After installation, open your terminal and run:

```shell
ollama run llama3.2
```

That's it! You're now chatting with a powerful LLM running on your own hardware.

Or see the [quick start guide](./getting_started/quickstart.md) for more details.

## Documentation

The Ollama documentation provides comprehensive guides and references

```{toctree}
:maxdepth: 3

getting_started/quickstart
getting_started/index
installation/index.md
api
devs/index.md
resources/index.md
```

