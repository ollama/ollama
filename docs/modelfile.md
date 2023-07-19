# Ollama Model File

A Model file is the blueprint to create and share models with Ollama.

## Format

The format of the Modelfile:

```modelfile
# comment
INSTRUCTION arguments
```

| Instruction              | Description                                              |
|------------------------- |--------------------------------------------------------- |
| FROM<br>(required)       | Defines the base model to be used when creating a model  |
| PARAMETER<br>(optional)  | Sets the parameters for how the model will be run        |
| PROMPT <br>(optional)    | Sets the prompt to use when the model will be run        |
| LICENSE<br>(optional)    | Specify the license of the model                         |

## Examples

An example of a model file creating a mario blueprint:

```
FROM llama2
PARAMETER temperature 1
PROMPT """
System: You are Mario from super mario bros, acting as an assistant.
User: {{ .Prompt }}
Assistant:
"""
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

The PARAMETER instruction defines a parameter that can be set when the model is run.

```
PARAMETER <parameter> <parametervalue>
```

### Valid Parameters and Values

| Parameter     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Value Type | Example Usage     |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------|
| NumCtx        | Sets the size of the prompt context size length model.  (Default: 2048)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | int        | Numctx 4096       |
| temperature   | The temperature of the model. Higher temperatures result in more creativity in the response (Default: 0.8)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | float      | Temperature 0.7   |
| TopK          | Top-k sampling is a text generation method that selects the next token only from the top k most likely tokens predicted by the model. It helps reduce the risk of generating low-probability or nonsensical tokens, but it may also limit the diversity of the output. A higher value for top-k (e.g., 100) will consider more tokens and lead to more diverse text, while a lower value (e.g., 10) will focus on the most probable tokens and generate more conservative text. (Default: 40)                                                                                              | int        | TopK 40           |
| TopP          | Top-p sampling (nucleus sampling), is a text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top-p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)                                                                                 | float      | TopP 0.9          |
| NumGPU        | The number of GPUs to use. On macOS it defaults to 1 to enable metal support, 0 to disable.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | int        | numGPU 1          |
| RepeatLastN   | Sets the number of tokens in the history to consider for penalizing repetition. A larger value will look further back in the generated text to prevent repetitions, while a smaller value will only consider recent tokens. A value of 0 disables the penalty, and a value of -1 sets the number of tokens considered equal to the context size.  (Default: 64, 0 = disabled, -1 = ctx-size)                                                                                                                                                                                               | int        | RepeatLastN 64    |
| RepeatPenalty | Helps prevent the model from generating repetitive or monotonous text. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)                                                                                                                                                                                                                                                                                                                                                                            | float      | RepeatPenalty 1.1 |
| TFSZ          | Tail free sampling. It is a text generation technique that aims to reduce the impact of less likely tokens, which may be less relevant, less coherent, or nonsensical, on the output. The method adjusts the logits (token probabilities) by raising them to the power of the parameter z (set value). A higher value of z (e.g., 2.0) will further suppress less likely tokens from the tail of the distribution, while a value of 1.0 disables the effect of TFS. By setting the parameter z, you can control how much the probabilities of less likely tokens are reduced. (default: 1) | float      | TFSZ 1            |
| Mirostat      | Enable Mirostat sampling, controlling perplexity during text generation. Mirostat is an algorithm that actively maintains the quality of generated text within a desired range during text generation. It aims to strike a balance between coherence and diversity, avoiding low-quality output caused by excessive repetition (boredom traps) or incoherence (confusion traps).  (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)                                                                                                                                               | int        | Mirostat 0        |
| MirostatTau   | Sets the Mirostat target entropy, which represents the desired perplexity value for the generated text. Adjusting the target entropy allows you to control the balance between coherence and diversity in the generated text. A lower value will result in more focused and coherent text, while a higher value will lead to more diverse and potentially less coherent text. (Default: 5.0)                                                                                                                                                                                               | float      | MirostatTau 5.0   |
| MirostatEta   | Sets the Microstat learning rate, which influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)                                                                                                                                                                                                                                                                                                                   | float      | MirostatEta 0.1   |
| NumThread     | Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores).                                                                                                                                                                                                                                                                                                                                    | int        | NumThread 8       |

## PROMPT

Prompt is a set of instructions to an LLM to cause the model to return desired response(s). Typically there are 3-4 components to a prompt: System, context, user, and response.

```modelfile
PROMPT """
{{- if not .Context }}
### System:
You are a content marketer who needs to come up with a short but succinct tweet. Make sure to include the appropriate hashtags and links. Sometimes when appropriate, describe a meme that can be includes as well. All answers should be in the form of a tweet which has a max size of 280 characters. Every instruction will be the topic to create a tweet about.
{{- end }}
### Instruction:
{{ .Prompt }}

### Response:
"""

```

## Notes

- the **modelfile is not case sensitive**. In the examples, we use uppercase for instructions to make it easier to distinguish it from arguments.
- Instructions can be in any order. In the examples, we start with FROM instruction to keep it easily readable.
