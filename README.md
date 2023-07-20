<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" height="200px" srcset="https://github.com/jmorganca/ollama/assets/3325447/318048d2-b2dd-459c-925a-ac8449d5f02c">
    <img alt="logo" height="200px" src="https://github.com/jmorganca/ollama/assets/3325447/c7d6e15f-7f4d-4776-b568-c084afa297c2">
  </picture>
</div>

# Ollama

[![Discord](https://dcbadge.vercel.app/api/server/ollama?style=flat&compact=true)](https://discord.gg/ollama)

Create, run, and share large language models (LLMs). Ollama bundles a model’s weights, configuration, prompts, and more into self-contained packages that can run on any machine.

> Note: Ollama is in early preview. Please report any issues you find.

## Download

- [Download](https://ollama.ai/download) for macOS on Apple Silicon (Intel coming soon)
- Download for Windows and Linux (coming soon)
- Build [from source](#building)

## Quickstart

To run and chat with [Llama 2](https://ai.meta.com/llama), the new model by Meta:

```
ollama run llama2
```

## Model library

Ollama includes a library of open-source, pre-trained models. More models are coming soon. You should have at least 8 GB of RAM to run the 3B models, 16 GB to run the 7B models, and 32 GB to run the 13B models.

| Model                    | Parameters | Size  | Download                    |
| ------------------------ | ---------- | ----- | --------------------------- |
| Llama2                   | 7B         | 3.8GB | `ollama pull llama2`        |
| Llama2 13B               | 13B        | 7.3GB | `ollama pull llama2:13b`    |
| Orca Mini                | 3B         | 1.9GB | `ollama pull orca`          |
| Vicuna                   | 7B         | 3.8GB | `ollama pull vicuna`        |
| Nous-Hermes              | 13B        | 7.3GB | `ollama pull nous-hermes`   |
| Wizard Vicuna Uncensored | 13B        | 7.3GB | `ollama pull wizard-vicuna` |

## Examples

### Run a model

```
ollama run llama2
>>> hi
Hello! How can I help you today?
```

### Create a custom character model

Pull a base model:

```
ollama pull orca
```

Create a `Modelfile`:

```
FROM orca
PROMPT """
### System:
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.

### User:
{{ .Prompt }}

### Response:
"""
```

Next, create and run the model:

```
ollama create mario -f ./Modelfile
ollama run mario
>>> hi
Hello! It's your friend Mario.
```

For more examples, see the [examples](./examples) directory.

### Pull a model from the registry

```
ollama pull nous-hermes
```

## Building

```
go build .
```

To run it start the server:

```
./ollama serve &
```

Finally, run a model!

```
./ollama run llama2
```
