# Ollama

Run ai models locally.

> _Note: this project is a work in progress. The features below are still in development_

**Features**

- Run models locally on macOS (Windows, Linux and other platforms coming soon)
- Ollama uses the fastest loader available for your platform and model (e.g. llama.cpp, Core ML and other loaders coming soon)
- Import models from local files
- Find and download models on Hugging Face and other sources (coming soon)
- Support for running and switching between multiple models at a time (coming soon)
- Native desktop experience (coming soon)
- Built-in memory (coming soon)

## Install

```
pip install ollama
```

## Quickstart

```
% ollama run huggingface.co/TheBloke/orca_mini_3B-GGML
Pulling huggingface.co/TheBloke/orca_mini_3B-GGML...
Downloading [================>          ] 66.67% (2/3) 30.2MB/s

...
...
...

> Hello

Hello, how may I help you?
```

## Python SDK

### Example

```python
import ollama
ollama.generate("orca-mini-3b", "hi")
```

### `ollama.generate(model, message)`

Generate a completion

```python
ollama.generate("./llama-7b-ggml.bin", "hi")
```

### `ollama.models()`

List available local models

```python
models = ollama.models()
```

### `ollama.serve()`

Serve the ollama http server

```
ollama.serve()
```

### `ollama.add(filepath)`

Add a model by importing from a file

```python
ollama.add("./path/to/model")
```

### `ollama.load(model)`

Manually a model for generation

```python
ollama.load("model")
```

### `ollama.unload(model)`

Unload a model

```python
ollama.unload("model")
```

### `ollama.pull(model)`

Download a model

```python
ollama.pull("huggingface.co/thebloke/llama-7b-ggml")
```

## Cooming Soon

### `ollama.search("query")`

Search for compatible models that Ollama can run

```python
ollama.search("llama-7b")
```

## Documentation

- [Development](docs/development.md)
