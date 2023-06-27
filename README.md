# Ollama

- Run models easily
- Download, manage and import models

## Install

```
pip install ollama
```

## Example quickstart

```python
import ollama
ollama.generate("./llama-7b-ggml.bin", "hi")
```

## Reference

### `ollama.load`

Load a model for generation

```python
ollama.load("model name")
```

### `ollama.generate("message")`

Generate a completion

```python
ollama.generate(model, "hi")
```

### `ollama.models`

List available local models

```
models = ollama.models()
```

### `ollama.serve`

Serve the ollama http server

## Cooing Soon

### `ollama.pull`

Download a model

```python
ollama.pull("huggingface.co/thebloke/llama-7b-ggml")
```

### `ollama.import`

Import a model from a file

```python
ollama.import("./path/to/model")
```

### `ollama.search`

Search for compatible models that Ollama can run

```python
ollama.search("llama-7b")
```

## Future CLI

In the future, there will be an easy CLI for testing out models

```
ollama run huggingface.co/thebloke/llama-7b-ggml
> Downloading [================>          ] 66.67% (2/3) 30.2MB/s
```
