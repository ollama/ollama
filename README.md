# Ollama

Ollama is a tool for running large language models. It's designed to be easy to use and fast.

> _Note: this project is a work in progress. Certain models that can be run with `ollama` are intended for research and/or non-commercial use only._

## Install

Using `pip`:

```
pip install ollama
```

Using `docker`:

```
docker run ollama/ollama
```

## Quickstart

To run a model, use `ollama run`:

```
ollama run orca-mini-3b
```

You can also run models from hugging face:

```
ollama run huggingface.co/TheBloke/orca_mini_3B-GGML
```

Or directly via downloaded model files:

```
ollama run ~/Downloads/orca-mini-13b.ggmlv3.q4_0.bin
```

## Documentation

- [Development](docs/development.md)
- [Python SDK](docs/python.md)
