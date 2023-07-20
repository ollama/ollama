<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" height="200px" srcset="https://github.com/jmorganca/ollama/assets/3325447/56ea1849-1284-4645-8970-956de6e51c3c">
    <img alt="logo" height="200px" src="https://github.com/jmorganca/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
  </picture>
</div>

# Ollama

[![Discord](https://dcbadge.vercel.app/api/server/ollama?style=flat&compact=true)](https://discord.gg/ollama)

> Note: Ollama is in early preview. Please report any issues you find.

Create, run, and share portable large language models (LLMs). Ollama bundles a model’s weights, configuration, prompts, and more into self-contained packages that can run on any machine.

### Portable Large Language Models (LLMs)

Package models as a series of layers in a portable, easy to manage format.

#### The idea behind Ollama

- Universal model format that can run anywhere: desktop, cloud servers & other devices.
- Encapsulate everything a model needs to operate – weights, configuration, and data – into a single package.
- Build custom models from base models like Meta's [Llama 2](https://ai.meta.com/llama/)
- Share large models without having to transmit large amounts of data.

<picture>
  <source media="(prefers-color-scheme: dark)" height="480" srcset="https://github.com/jmorganca/ollama/assets/251292/2e05cf23-e3c6-403e-9910-3d622801f4b8">
  <img alt="logo" height="480" src="https://github.com/jmorganca/ollama/assets/251292/2e05cf23-e3c6-403e-9910-3d622801f4b8">
</picture>

This format is inspired by the [image spec](https://github.com/opencontainers/image-spec) originally introduced by Docker for Linux containers. Ollama extends this format to package large language models.

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
ollama pull llama2
```

Create a `Modelfile`:

```
FROM llama2

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system prompt
SYSTEM """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
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
ollama pull orca
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
