# Ollama

![Ollama Logo](https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7)

## Run Large Language Models Locally

Ollama lets you run powerful large language models (LLMs) on your own hardware. Get started with AI without sending your data to third-party services.

## Key Features

- **Run models locally** on your Mac, Windows, or Linux machine
- **No cloud required** - your data stays on your device
- **Easy to use** - simple commands to download and run models
- **Customizable** - create and modify your own models
- **API access** - integrate with your applications

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

## Documentation

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

getting_started/index
getting_started/quickstart
getting_started/examples
getting_started/import
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Installation

installation/index
installation/linux
installation/windows
installation/docker
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

reference/index
reference/api
reference/modelfile
reference/openai
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Resources

resources/index
resources/troubleshooting
resources/faq
resources/development
resources/benchmark
resources/gpu
resources/template
```

## Community

Join our community to get help, share your experiences, and contribute to Ollama:

- [Discord](https://discord.gg/ollama)
- [Reddit](https://reddit.com/r/ollama)
- [GitHub](https://github.com/ollama/ollama)
