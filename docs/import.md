# Import

GGUF models and select Safetensors models can be imported directly into Ollama.

## Import GGUF

A binary GGUF file can be imported directly into Ollama through a Modelfile.

```dockerfile
FROM /path/to/file.gguf
```

## Import Safetensors

If the model being imported is one of these architectures, it can be imported directly into Ollama through a Modelfile:

 - LlamaForCausalLM
 - MistralForCausalLM
 - GemmaForCausalLM

```dockerfile
FROM /path/to/safetensors/directory
```

For architectures not directly convertable by Ollama, see llama.cpp's [guide](https://github.com/ggerganov/llama.cpp/blob/master/README.md#prepare-and-quantize) on conversion. After conversion, see [Import GGUF](#import-gguf).

## Automatic Quantization

> [!NOTE]
> Automatic quantization requires v0.1.35 or higher.

Ollama is capable of quantizing FP16 or FP32 models to any of the supported quantizations with the `-q/--quantize` flag in `ollama create`.

```dockerfile
FROM /path/to/my/gemma/f16/model
```

```shell
$ ollama create -q Q4_K_M mymodel
transferring model data
quantizing F16 model to Q4_K_M
creating new layer sha256:735e246cc1abfd06e9cdcf95504d6789a6cd1ad7577108a70d9902fef503c1bd
creating new layer sha256:0853f0ad24e5865173bbf9ffcc7b0f5d56b66fd690ab1009867e45e7d2c4db0f
writing manifest
success
```

### Supported Quantizations

<details>
<summary>Legacy Quantization</summary>

- `Q4_0`
- `Q4_1`
- `Q5_0`
- `Q5_1`
- `Q8_0`

</details>

<details>
<summary>K-means Quantization</summary>`

- `Q3_K_S`
- `Q3_K_M`
- `Q3_K_L`
- `Q4_K_S`
- `Q4_K_M`
- `Q5_K_S`
- `Q5_K_M`
- `Q6_K`

</details>

> [!NOTE]
> Activation-aware Weight Quantization (i.e. IQ) are not currently supported for automatic quantization however you can still import the quantized model into Ollama, see [Import GGUF](#import-gguf).

## Template Detection

> [!NOTE]
> Template detection requires v0.1.42 or higher.

Ollama uses model metadata, specifically `tokenizer.chat_template`, to automatically create a template appropriate for the model you're importing.

```dockerfile
FROM /path/to/my/gemma/model
```

```shell
$ ollama create mymodel
transferring model data
using autodetected template gemma-instruct
creating new layer sha256:baa2a0edc27d19cc6b7537578a9a7ba1a4e3214dc185ed5ae43692b319af7b84
creating new layer sha256:ba66c3309914dbef07e5149a648fd1877f030d337a4f240d444ea335008943cb
writing manifest
success
```

Defining a template in the Modelfile will disable this feature which may be useful if you want to use a different template than the autodetected one.
