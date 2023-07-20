# Ollama Model File

A model file is the blueprint to create and share models with Ollama.

## Format

The format of the Modelfile:

```modelfile
# comment
INSTRUCTION arguments
```

| Instruction               | Description                                           |
| ------------------------- | ----------------------------------------------------- |
| `FROM`<br>(required)      | Defines the base model to use                         |
| `PARAMETER`<br>(optional) | Sets the parameters for how Ollama will run the model |
| `SYSTEM`<br>(optional)    | Specifies the system prompt that will set the context |
| `TEMPLATE`<br>(optional)  | The full prompt template to be sent to the model      |
| `LICENSE`<br>(optional)   | Specifies the legal license                           |

## Examples

An example of a model file creating a mario blueprint:

```
FROM llama2
# sets the temperature to 1 [higher is more creative, lower is more coherent]
# sets the context size to 4096
PARAMETER temperature 1
PARAMETER num_ctx 4096

# Check for first system message, so the model output won't repeat itself.
# <<SYS>> and [INST] are special tags used by the Llama2 model.

PROMPT """
{{- if .First }}
<<SYS>>
You are Mario from super mario bros, acting as an assistant.
<</SYS>>

{{- end }}
[INST] {{ .Prompt }} [/INST]
"""
```

To use this:

1. Save it as a file (eg. modelfile)
2. `ollama create NAME -f <location of the file eg. ./modelfile>'`
3. `ollama run NAME`
4. Start using the model!

## FROM (Required)

The FROM instruction Defines the base model to use when creating a model.

```
FROM <model name>:<tag>
```

### Build from llama2

```
FROM llama2:latest
```

A list of available base models:
<https://github.com/jmorganca/ollama#model-library>

### Build from a bin file

```
FROM ./ollama-model.bin
```

## PARAMETER (Optional)

The `PARAMETER` instruction defines a parameter that can be set when the model is run.

```
PARAMETER <parameter> <parametervalue>
```

### Valid Parameters and Values

| Parameter      | Description                                                                                                                                                                                                                                             | Value Type | Example Usage      |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------ |
| num_ctx        | Sets the size of the prompt context size length model. (Default: 2048)                                                                                                                                                                                  | int        | num_ctx 4096       |
| temperature    | The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)                                                                                                                                     | float      | temperature 0.7    |
| top_k          | Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)                                                                        | int        | top_k 40           |
| top_p          | Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)                                                                 | float      | top_p 0.9          |
| num_gpu        | The number of GPUs to use. On macOS it defaults to 1 to enable metal support, 0 to disable.                                                                                                                                                             | int        | num_gpu 1          |
| repeat_last_n  | Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = ctx-size)                                                                                                                                          | int        | repeat_last_n 64   |
| repeat_penalty | Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)                                                                     | float      | repeat_penalty 1.1 |
| tfs_z          | Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)                                               | float      | tfs_z 1            |
| mirostat       | Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)                                                                                                                                         | int        | mirostat 0         |
| mirostat_tau   | Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. (Default: 5.0)                                                                                                         | float      | mirostat_tau 5.0   |
| mirostat_eta   | Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)                        | float      | mirostat_eta 0.1   |
| num_thread     | Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores). | int        | num_thread 8       |

## Prompt

When building on top of the base models supplied by Ollama, it comes with the prompt template predefined. To override the supplied system prompt, simply add `SYSTEM: insert system prompt` to change the systme prompt.

### Prompt Template

`TEMPLATE` the full prompt template to be passed into the model. It may include (optionally) a system prompt, user prompt, and assistant prompt. This is used to create a full custom prompt, and syntax may be model specific.

## Notes

- the **modelfile is not case sensitive**. In the examples, we use uppercase for instructions to make it easier to distinguish it from arguments.
- Instructions can be in any order. In the examples, we start with FROM instruction to keep it easily readable.
