# Ollama Model File Reference

Ollama can build models automatically by reading the instructions from a Modelfile. A Modelfile is a text document that represents the complete configuration of the Model. You can see that a Modelfile is very similar to a Dockerfile.

## Format

Here is the format of the Modelfile:

```modelfile
# comment
INSTRUCTION arguments
```

Nothing in the file is case-sensitive. However, the convention is for instructions to be uppercase to make it easier to distinguish from the arguments.

A Modelfile can include instructions in any order. But the convention is to start the Modelfile with the FROM instruction.

Although the example above shows a comment starting with a hash character, any instruction that is not recognized is seen as a comment. 

## FROM

```modelfile
FROM <image>[:<tag>]
```

This defines the base model to be used. An image can be a known image on the Ollama Hub, or a fully-qualified path to a model file on your system

## LICENSE

```modelfile
LICENSE """
<license text>
"""
```

Some models need to be distributed with a license agreement. For example, the distribution clause for the Llama2 license requires including the license with the model. 

## PARAMETER

```modelfile
PARAMETER <parameter> <parametervalue>
```

The PARAMETER instruction defines a parameter that can be set when the model is run.

### Valid Parameters and Values

| Parameter        | Description                                                                                 | Value Type | Value Range |
| ---------------- | ------------------------------------------------------------------------------------------- | ---------- | ----------- |
| NumCtx           |                                                                                             | int        |             |
| NumGPU           |                                                                                             | int        |             |
| MainGPU          |                                                                                             | int        |             |
| LowVRAM          |                                                                                             | bool       |             |
| F16KV            |                                                                                             | bool       |             |
| LogitsAll        |                                                                                             | bool       |             |
| VocabOnly        |                                                                                             | bool       |             |
| UseMMap          |                                                                                             | bool       |             |
| EmbeddingOnly    |                                                                                             | bool       |             |
| RepeatLastN      |                                                                                             | int        |             |
| RepeatPenalty    |                                                                                             | float      |             |
| FrequencyPenalty |                                                                                             | float      |             |
| PresencePenalty  |                                                                                             | float      |             |
| temperature      | The temperature of the model. Higher temperatures result in more creativity in the response | float      | 0 - 1       |
| TopK             |                                                                                             | int        |             |
| TopP             |                                                                                             | float      |             |
| TFSZ             |                                                                                             | float      |             |
| TypicalP         |                                                                                             | float      |             |
| Mirostat         |                                                                                             | int        |             |
| MirostatTau      |                                                                                             | float      |             |
| MirostatEta      |                                                                                             | float      |             |
| NumThread        |                                                                                             | int |             |


## PROMPT

Prompt is a multiline instruction that defines the prompt to be used when the model is run. Typically there are 3-4 components to a prompt: System, context, user, and response.

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