# Modelfile Reference

> [!NOTE]
> `Modelfile` syntax is in development

A Modelfile is the blueprint to create and share models with Ollama. This page provides an overview of the Modelfile format and instructions. For detailed information, please refer to the [full Modelfile documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md).

## Format

The format of the `Modelfile`:

```
# comment
INSTRUCTION arguments
```

| Instruction                        | Description                                                    |
|------------------------------------| -------------------------------------------------------------- |
| [`FROM`](#from-required) (required) | Defines the base model to use.                                 |
| [`PARAMETER`](#parameter)          | Sets the parameters for how Ollama will run the model.         |
| [`TEMPLATE`](#template)            | The full prompt template to be sent to the model.              |
| [`SYSTEM`](#system)                | Specifies the system message that will be set in the template. |
| [`ADAPTER`](#adapter)              | Defines the (Q)LoRA adapters to apply to the model.            |
| [`LICENSE`](#license)              | Specifies the legal license.                                   |
| [`MESSAGE`](#message)              | Specify message history.                                       |

## Basic Example

An example of a `Modelfile` creating a mario blueprint:

```
FROM llama3.2
# sets the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1
# sets the context window size to 4096, this controls how many tokens the LLM can use as context to generate the next token
PARAMETER num_ctx 4096

# sets a custom system message to specify the behavior of the chat assistant
SYSTEM You are Mario from super mario bros, acting as an assistant.
```

To use this:

1. Save it as a file (e.g. `Modelfile`)
2. `ollama create choose-a-model-name -f <location of the file e.g. ./Modelfile>`
3. `ollama run choose-a-model-name`
4. Start using the model!

To view the Modelfile of a given model, use the `ollama show --modelfile` command.

## Instructions

### FROM (Required)

The `FROM` instruction defines the base model to use when creating a model.

```
FROM <model name>:<tag>
```

You can build from:
- An existing model: `FROM llama3.2`
- A Safetensors model: `FROM /path/to/safetensors/directory`
- A GGUF file: `FROM /path/to/file.gguf`

### PARAMETER

The `PARAMETER` instruction sets the parameters for how Ollama will run the model.

```
PARAMETER <name> <value>
```

#### Valid Parameters and Values

| Parameter | Description |
| --- | --- |
| `mirostat` | Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0) |
| `mirostat_eta` | Mirostat learning rate, parameter eta in the paper. (default: 0.1) |
| `mirostat_tau` | Mirostat target entropy, parameter tau in the paper. (default: 5.0) |
| `num_ctx` | Sets the size of the context window used to generate the next token. (default: 2048) |
| `num_gpu` | The number of layers to send to the GPU(s). On macOS it defaults to 1 to enable metal support, 0 to disable. |
| `num_thread` | Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores). |
| `repeat_last_n` | Sets how far back for the model to look back to prevent repetition. (default: 64, 0 = disabled, -1 = num_ctx) |
| `repeat_penalty` | Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (default: 1.1) |
| `temperature` | The temperature of the model. Increasing the temperature will make the model answer more creatively. (default: 0.8) |
| `seed` | Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (default: 0) |
| `stop` | Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return. Multiple stop patterns may be set by specifying multiple separate `PARAMETER stop <pattern>` lines in the Modelfile. |
| `tfs_z` | Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1) |
| `num_predict` | Maximum number of tokens to predict when generating text. (default: 128, -1 = infinite generation, -2 = fill context) |
| `top_k` | Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (default: 40) |
| `top_p` | Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (default: 0.9) |

For more details on parameters, see the [full Modelfile documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).

### TEMPLATE

The `TEMPLATE` instruction specifies the full prompt template to be sent to the model.

```
TEMPLATE """<template>"""
```

### SYSTEM

The `SYSTEM` instruction specifies the system message that will be set in the template.

```
SYSTEM """<system message>"""
```

### ADAPTER

The `ADAPTER` instruction defines the (Q)LoRA adapters to apply to the model.

```
ADAPTER <path to adapter>
```

### LICENSE

The `LICENSE` instruction specifies the legal license for the model.

```
LICENSE """<license text>"""
```

### MESSAGE

The `MESSAGE` instruction specifies message history.

```
MESSAGE <role> """<content>"""
```

## Further Reading

For more detailed information about Modelfiles, including advanced usage and examples, please refer to the [full Modelfile documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md).
