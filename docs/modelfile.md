# Ollama Model File

> Note: this model file syntax is in development

A model file is the blueprint to create and share models with Ollama.

## Table of Contents

- [Format](#format)
- [Examples](#examples)
- [Instructions](#instructions)
  - [FROM (Required)](#from-required)
    - [Build from llama2](#build-from-llama2)
    - [Build from a bin file](#build-from-a-bin-file)
  - [EMBED](#embed)
  - [PARAMETER](#parameter)
    - [Valid Parameters and Values](#valid-parameters-and-values)
  - [TEMPLATE](#template)
    - [Template Variables](#template-variables)
  - [SYSTEM](#system)
  - [ADAPTER](#adapter)
  - [LICENSE](#license)
- [Notes](#notes)

## Format

The format of the Modelfile:

```modelfile
# comment
INSTRUCTION arguments
```

| Instruction                         | Description                                                   |
| ----------------------------------- | ------------------------------------------------------------- |
| [`FROM`](#from-required) (required) | Defines the base model to use.                                |
| [`PARAMETER`](#parameter)           | Sets the parameters for how Ollama will run the model.        |
| [`TEMPLATE`](#template)             | The full prompt template to be sent to the model.             |
| [`SYSTEM`](#system)                 | Specifies the system prompt that will be set in the template. |
| [`ADAPTER`](#adapter)               | Defines the (Q)LoRA adapters to apply to the model.           |
| [`LICENSE`](#license)               | Specifies the legal license.                                  |

## Examples

An example of a model file creating a mario blueprint:

```
FROM llama2
# sets the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1
# sets the context window size to 4096, this controls how many tokens the LLM can use as context to generate the next token
PARAMETER num_ctx 4096

# sets a custom system prompt to specify the behavior of the chat assistant
SYSTEM You are Mario from super mario bros, acting as an assistant.
```

To use this:

1. Save it as a file (eg. `Modelfile`)
2. `ollama create NAME -f <location of the file eg. ./Modelfile>'`
3. `ollama run NAME`
4. Start using the model!

More examples are available in the [examples directory](../examples).

## Instructions

### FROM (Required)

The FROM instruction defines the base model to use when creating a model.

```
FROM <model name>:<tag>
```

#### Build from llama2

```
FROM llama2
```

A list of available base models:
<https://github.com/jmorganca/ollama#model-library>

#### Build from a bin file

```
FROM ./ollama-model.bin
```

This bin file location should be specified as an absolute path or relative to the Modelfile location.

### EMBED

The EMBED instruction is used to add embeddings of files to a model. This is useful for adding custom data that the model can reference when generating an answer. Note that currently only text files are supported, formatted with each line as one embedding.

```
FROM <model name>:<tag>
EMBED <file path>.txt
EMBED <different file path>.txt
EMBED <path to directory>/*.txt
```

### PARAMETER

The `PARAMETER` instruction defines a parameter that can be set when the model is run.

```
PARAMETER <parameter> <parametervalue>
```

### Valid Parameters and Values

| Parameter      | Description                                                                                                                                                                                                                                             | Value Type | Example Usage        |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------------------- |
| mirostat       | Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)                                                                                                                                         | int        | mirostat 0           |
| mirostat_eta   | Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)                        | float      | mirostat_eta 0.1     |
| mirostat_tau   | Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. (Default: 5.0)                                                                                                         | float      | mirostat_tau 5.0     |
| num_ctx        | Sets the size of the context window used to generate the next token. (Default: 2048)                                                                                                                                                                    | int        | num_ctx 4096         |
| num_gqa        | The number of GQA groups in the transformer layer. Required for some models, for example it is 8 for llama2:70b                                                                                                                                         | int        | num_gqa 1            |
| num_gpu        | The number of layers to send to the GPU(s). On macOS it defaults to 1 to enable metal support, 0 to disable.                                                                                                                                            | int        | num_gpu 50           |
| num_thread     | Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores). | int        | num_thread 8         |
| repeat_last_n  | Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)                                                                                                                                           | int        | repeat_last_n 64     |
| repeat_penalty | Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)                                                                     | float      | repeat_penalty 1.1   |
| temperature    | The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)                                                                                                                                     | float      | temperature 0.7      |
| stop           | Sets the stop sequences to use.                                                                                                                                                                                                                         | string     | stop "AI assistant:" |
| tfs_z          | Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)                                               | float      | tfs_z 1              |
| num_predict    | Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)                                                                                                                                   | int        | num_predict 42       |
| top_k          | Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)                                                                        | int        | top_k 40             |
| top_p          | Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)                                                                 | float      | top_p 0.9            |

### TEMPLATE

`TEMPLATE` of the full prompt template to be passed into the model. It may include (optionally) a system prompt and a user's prompt. This is used to create a full custom prompt, and syntax may be model specific.

#### Template Variables

| Variable        | Description                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| `{{ .System }}` | The system prompt used to specify custom behavior, this must also be set in the Modelfile as an instruction. |
| `{{ .Prompt }}` | The incoming prompt, this is not specified in the model file and will be set based on input.                 |
| `{{ .First }}`  | A boolean value used to render specific template information for the first generation of a session.          |

```
TEMPLATE """
{{- if .First }}
### System:
{{ .System }}
{{- end }}

### User:
{{ .Prompt }}

### Response:
"""

SYSTEM """<system message>"""
```

### SYSTEM

The `SYSTEM` instruction specifies the system prompt to be used in the template, if applicable.

```
SYSTEM """<system message>"""
```

### ADAPTER

The `ADAPTER` instruction specifies the LoRA adapter to apply to the base model. The value of this instruction should be an absolute path or a path relative to the Modelfile and the file must be in a GGML file format. The adapter should be tuned from the base model otherwise the behaviour is undefined.

```
ADAPTER ./ollama-lora.bin
```

### LICENSE

The `LICENSE` instruction allows you to specify the legal license under which the model used with this Modelfile is shared or distributed.

```
LICENSE """
<license text>
"""
```

## Notes

- the **modelfile is not case sensitive**. In the examples, we use uppercase for instructions to make it easier to distinguish it from arguments.
- Instructions can be in any order. In the examples, we start with FROM instruction to keep it easily readable.
