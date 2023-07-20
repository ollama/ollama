# Ollama Model File

A model file is the blueprint to create and share models with Ollama.

## Format

The format of the Modelfile:

```modelfile
# comment
INSTRUCTION arguments
```

| Instruction               | Description                                              |
|-------------------------- |--------------------------------------------------------- |
| `FROM`<br>(required)      | Defines the base model to be used when creating a model  |
| `PARAMETER`<br>(optional) | Sets the parameters for how the model will be run        |
| `TEMPLATE`<br>(optional)  | Sets the prompt template to use when the model will be run        |
| `SYSTEM`<br>(optional)    | // todo |
| `LICENSE`<br>(optional)   | Specify the license of the model. It is additive, and                          |

## Examples

An example of a model file creating a mario blueprint:

```
FROM llama2
PARAMETER temperature 1
TEMPLATE """
System: {{ .System }}
User: {{ .Prompt }}
Assistant:
"""

SYSTEM You are Mario from super mario bros, acting as an assistant.
```

To use this:

1. Save it as a file (eg. modelfile)
2. `ollama create NAME -f <location of the file eg. ./modelfile>'`
3. `ollama run NAME`
4. Start using the model!

## FROM (Required)

The FROM instruction defines the base model to be used when creating a model.

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

| Parameter     | Description                                                                                                                                                                                                                                             | Value Type | Example Usage     |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------|
| NumCtx        | Sets the size of the prompt context size length model. (Default: 2048)                                                                                                                                                                                  | int        | Numctx 4096       |
| temperature   | The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)                                                                                                                                     | float      | Temperature 0.7   |
| TopK          | Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)                                                                        | int        | TopK 40           |
| TopP          | Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)                                                                 | float      | TopP 0.9          |
| NumGPU        | The number of GPUs to use. On macOS it defaults to 1 to enable metal support, 0 to disable.                                                                                                                                                             | int        | numGPU 1          |
| RepeatLastN   | Sets how far back for the model to look back to prevent repetition.  (Default: 64, 0 = disabled, -1 = ctx-size)                                                                                                                                         | int        | RepeatLastN 64    |
| RepeatPenalty | Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)                                                                     | float      | RepeatPenalty 1.1 |
| TFSZ          | Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)                                               | float      | TFSZ 1            |
| Mirostat      | Enable Mirostat sampling for controlling perplexity.  (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)                                                                                                                                        | int        | Mirostat 0        |
| MirostatTau   | Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. (Default: 5.0)                                                                                                         | float      | MirostatTau 5.0   |
| MirostatEta   | Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)                        | float      | MirostatEta 0.1   |
| NumThread     | Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores). | int        | NumThread 8       |

## TEMPLATE

`TEMPLATE` is a set of instructions to an LLM to cause the model to return desired response(s). Typically there are 3-4 components to a prompt: system, input, and response.

```modelfile
TEMPLATE """
### System:
{{ .System }}

### Instruction:
{{ .Prompt }}

### Response:
"""

SYSTEM """
You are a content marketer who needs to come up with a short but succinct tweet. Make sure to include the appropriate hashtags and links. Sometimes when appropriate, describe a meme that can be includes as well. All answers should be in the form of a tweet which has a max size of 280 characters. Every instruction will be the topic to create a tweet about.
"""

```

## SYSTEM

// todo

## Notes

- the **modelfile is not case sensitive**. In the examples, we use uppercase for instructions to make it easier to distinguish it from arguments.
- Instructions can be in any order. In the examples, we start with FROM instruction to keep it easily readable.
