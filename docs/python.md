# Python SDK

## Install

```
pip install ollama
```

## Example

```python
import ollama
ollama.generate("orca-mini-3b", "hi")
```

## Reference

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

### `ollama.search(query)`

Search for compatible models that Ollama can run

```python
ollama.search("llama-7b")
```
