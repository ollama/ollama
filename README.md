# Ollama

- Run models, fast
- Download, manage and import models

## Install

```
pip install ollama
```

## Example quickstart

```python
import ollama
model_name = "huggingface.co/thebloke/llama-7b-ggml"
model = ollama.pull(model_name)
ollama.load(model)
ollama.generate(model_name, "hi")
```

## Reference

### `ollama.load`

Load a model from a path or a docker image

```python
ollama.load("model name")
```

### `ollama.generate("message")`

Generate a completion

```python
ollama.generate(model, "hi")
```

### `ollama.models`

List models

```
models = ollama.models()
```

### `ollama.serve`

Serve the ollama http server

## Cooing Soon

### `ollama.pull`

Examples:

```python
ollama.pull("huggingface.co/thebloke/llama-7b-ggml")
```

### `ollama.import`

Import an existing model into the model store

```python
ollama.import("./path/to/model")
```

### `ollama.search`

Search for compatible models that Ollama can run

```python
ollama.search("llama-7b")
```

## Future CLI

```
ollama run huggingface.co/thebloke/llama-7b-ggml
```
