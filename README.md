<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" height="200px" srcset="https://github.com/jmorganca/ollama/assets/3325447/318048d2-b2dd-459c-925a-ac8449d5f02c">
    <img alt="logo" height="200px" src="https://github.com/jmorganca/ollama/assets/3325447/c7d6e15f-7f4d-4776-b568-c084afa297c2">
  </picture>
</div>

# Ollama

Create, run, and share self-contained large language models (LLMs). Ollama bundles a model’s weights, configuration, prompts, and more into self-contained packages that run anywhere.

> Note: Ollama is in early preview. Please report any issues you find.

## Download

- [Download](https://ollama.ai/download) for macOS on Apple Silicon (Intel coming soon)
- Download for Windows and Linux (coming soon)
- Build [from source](#building)

## Examples

### Quickstart

```
ollama run llama2
>>> hi
Hello! How can I help you today?
```

### Creating a custom model

Create a `Modelfile`:

```
FROM llama2
PROMPT """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.

User: {{ .Prompt }}
Mario:
"""
```

Next, create and run the model:

```
ollama create mario -f ./Modelfile
ollama run mario
>>> hi
Hello! It's your friend Mario.
```

## Model library

Ollama includes a library of open-source, pre-trained models. More models are coming soon.

| Model                     | Parameters | Size  | Download                    |
| ----------------------    | ---------- | ----- | --------------------------- |
| Llama2                    | 7B         | 3.8GB | `ollama pull llama2`        |
| Orca Mini                 | 3B         | 1.9GB | `ollama pull orca`          |
| Vicuna                    | 7B         | 3.8GB | `ollama pull vicuna`        |
| Nous-Hermes               | 13B        | 7.3GB | `ollama pull nous-hermes`   |
| Wizard Vicuna Uncensored  | 13B        | 6.8GB | `ollama pull wizard-vicuna` |

## Building

```
go build .
```

To run it start the server:

```
./ollama server &
```

Finally, run a model!

```
./ollama run llama2
```
