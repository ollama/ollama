# Ollama

Ollama is a tool for running any large language model on any machine. It's designed to be easy to use and fast, supporting the largest number of models possible by using the fastest loader available for your platform and model.

> _Note: this project is a work in progress. Certain models that can be run with `ollama` are intended for research and/or non-commercial use only._

## Install

```
pip install ollama
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

## Coming Soon

### `ollama.search("query")`

Search for compatible models that Ollama can run

```python
ollama.search("llama-7b")
```

## Documentation

- [Development](docs/development.md)
