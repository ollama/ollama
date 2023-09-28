<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" height="200px" srcset="https://github.com/jmorganca/ollama/assets/3325447/56ea1849-1284-4645-8970-956de6e51c3c">
    <img alt="logo" height="200px" src="https://github.com/jmorganca/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
  </picture>
</div>

# Ollama

[![Discord](https://dcbadge.vercel.app/api/server/ollama?style=flat&compact=true)](https://discord.gg/ollama)

Get up and running with large language models locally.

### macOS

[Download](https://ollama.ai/download/Ollama-darwin.zip) 

### Linux & WSL2

```
curl https://ollama.ai/install.sh | sh
```

[Manual install instructions](https://github.com/jmorganca/ollama/blob/main/docs/linux.md)

### Windows 

coming soon

## Quickstart

To run and chat with [Llama 2](https://ollama.ai/library/llama2):

```
ollama run llama2
```

## Model library

Ollama supports a list of open-source models available on [ollama.ai/library](https://ollama.ai/library "ollama model library")

Here are some example open-source models that can be downloaded:

| Model              | Parameters | Size  | Download                       |
| ------------------ | ---------- | ----- | ------------------------------ |
| Mistral            | 7B         | 4.1GB | `ollama run mistral`           |
| Llama 2            | 7B         | 3.8GB | `ollama run llama2`            |
| Code Llama         | 7B         | 3.8GB | `ollama run codellama`         |
| Llama 2 Uncensored | 7B         | 3.8GB | `ollama run llama2-uncensored` |
| Llama 2 13B        | 13B        | 7.3GB | `ollama run llama2:13b`        |
| Llama 2 70B        | 70B        | 39GB  | `ollama run llama2:70b`        |
| Orca Mini          | 3B         | 1.9GB | `ollama run orca-mini`         |
| Vicuna             | 7B         | 3.8GB | `ollama run vicuna`            |

> Note: You should have at least 8 GB of RAM to run the 3B models, 16 GB to run the 7B models, and 32 GB to run the 13B models.

## Customize your own model

### Import from GGUF or GGML

Ollama supports importing GGUF and GGML file formats in the Modelfile. This means if you have a model that is not in the Ollama library, you can create it, iterate on it, and upload it to the Ollama library to share with others when you are ready.

1. Create a file named Modelfile, and add a `FROM` instruction with the local filepath to the model you want to import.

   ```
   FROM ./vicuna-33b.Q4_0.gguf
   ```

3. Create the model in Ollama

   ```
   ollama create name -f path_to_modelfile
   ```

5. Run the model

   ```
   ollama run name
   ```

### Customize a prompt

Models from the Ollama library can be customized with a prompt. The example

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

For more examples, see the [examples](./examples) directory. For more information on working with a Modelfile, see the [Modelfile](./docs/modelfile.md) documentation.

## CLI Reference

### Create a model

`ollama create` is used to create a model from a Modelfile.

### Pull a model

```
ollama pull llama2
```

> This command can also be used to update a local model. Only the diff will be pulled.

### Remove a model

```
ollama rm llama2
```

### Copy a model

```
ollama cp llama2 my-llama2
```

### Multiline input

For multiline input, you can wrap text with `"""`:

```
>>> """Hello,
... world!
... """
I'm a basic program that prints the famous "Hello, world!" message to the console.
```

### Pass in prompt as arguments

```
$ ollama run llama2 "summarize this file:" "$(cat README.md)"
 Ollama is a lightweight, extensible framework for building and running language models on the local machine. It provides a simple API for creating, running, and managing models, as well as a library of pre-built models that can be easily used in a variety of applications.
```

### List models on your computer

```
ollama list
```

### Start Ollama

`ollama serve` is used when you want to start ollama without running the desktop application.

## Building

Install `cmake` and `go`:

```
brew install cmake
brew install go
```

Then generate dependencies and build:

```
go generate ./...
go build .
```

Next, start the server:

```
./ollama serve
```

Finally, in a separate shell, run a model:

```
./ollama run llama2
```

## REST API

> See the [API documentation](./docs/api.md) for all endpoints.

Ollama has an API for running and managing models. For example to generate text from a model:

```
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt":"Why is the sky blue?"
}'
```

## Community Integrations

- [LangChain](https://python.langchain.com/docs/integrations/llms/ollama) and [LangChain.js](https://js.langchain.com/docs/modules/model_io/models/llms/integrations/ollama) with [example](https://js.langchain.com/docs/use_cases/question_answering/local_retrieval_qa)
- [LlamaIndex](https://gpt-index.readthedocs.io/en/stable/examples/llm/ollama.html)
- [Raycast extension](https://github.com/MassimilianoPasquini97/raycast_ollama)
- [Discollama](https://github.com/mxyng/discollama) (Discord bot inside the Ollama discord channel)
- [Continue](https://github.com/continuedev/continue)
- [Obsidian Ollama plugin](https://github.com/hinterdupfinger/obsidian-ollama)
- [Dagger Chatbot](https://github.com/samalba/dagger-chatbot)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [Discord AI Bot](https://github.com/mekb-turtle/discord-ai-bot)
- [HTML UI](https://github.com/rtcfirefly/ollama-ui)
- [Typescript UI](https://github.com/ollama-interface/Ollama-Gui?tab=readme-ov-file)
- [Dumbar](https://github.com/JerrySievert/Dumbar)
- [Emacs client](https://github.com/zweifisch/ollama)
