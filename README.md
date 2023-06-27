# Ollama

![header](https://github.com/jmorganca/ollama/assets/251292/98204129-c676-4ab5-be98-bcc45ba8e9e7)

The easiest way to run download and run ai models.

## Download

- [macOS](https://ollama.ai/download/darwin) (Apple Silicon)
- macOS (Intel – Coming soon)
- Windows (Coming soon)
- Linux (Coming soon)

## Python SDK

```
pip install ollama
```

### Python SDK quickstart

```python
import ollama
ollama.generate("./llama-7b-ggml.bin", "hi")
```

### `ollama.generate(model, message)`

Generate a completion

```python
ollama.generate("./llama-7b-ggml.bin", "hi")
```

### `ollama.load(model)`

Load a model for generation

```python
ollama.load("model")
```

### `ollama.models()`

List available local models

```
models = ollama.models()
```

### `ollama.serve()`

Serve the ollama http server

## Cooming Soon

### `ollama.pull(model)`

Download a model

```python
ollama.pull("huggingface.co/thebloke/llama-7b-ggml")
```

### `ollama.import(filename)`

Import a model from a file

```python
ollama.import("./path/to/model")
```

### `ollama.search("query")`

Search for compatible models that Ollama can run

```python
ollama.search("llama-7b")
```

## Future CLI

In the future, there will be an `ollama` CLI for running models on servers, in containers or for local development environments.

```
ollama generate huggingface.co/thebloke/llama-7b-ggml "hi"
> Downloading [================>          ] 66.67% (2/3) 30.2MB/s
```

## Documentation

- [Development](docs/development.md)
