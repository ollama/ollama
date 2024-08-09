
# Importing a model

You can import a model into Ollama:

  * From Safetensors weights; or
  * From a GGUF file

## Importing from Safetensors

Importing your model directly from the Safetensors weights is the easiest way to import a model. Ollama supports importing several different architectures including:

  * Llama (including Llama 2, Llama 3, and Llama 3.1);
  * Mistral (including Mistral 1, Mistral 2, and Mixtral); and
  * Gemma (including Gemma 1 and Gemma 2)

This includes importing either the foundation model as well as any fine tuned model which is based upon one of these architectures.


Importing a model requires three steps:

  1. Write a Modelfile
  2. Create your model with `ollama create my-model`
  3. Test your model with `ollama run my-model`


#### Writing a Modelfile

Create a Modelfile using a text editor which includes a `FROM` line which points to the directory with the model weights.

```dockerfile
FROM /path/to/safetensors/directory
```

If you create the Modelfile in the same directory as the weights, you can use the line `FROM .`.

You can include other settings in your Modelfile such as parameters for the model, the desired chat template, the system prompt, and any license information. Refer to the [Modelfile documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) for more information about Modelfile commands.

#### Create your model with `ollama create`

After you have created the Modelfile, use the command:

```bash
$ ollama create -f Modelfile <model name>
```

If the model architecture of your weights is supported, Ollama will take a few moments to import the weights. It will also attempt to find the correct _chat template_ inside the model's configuration data. You can override the template using the `TEMPLATE` command in your Modelfile.

#### Test your model with `ollama run`

Once you have created a model, you can run it using the command:

```bash
$ ollama run <model name>
```

This will load the newly created model into memory and you can test to make certain that it's working correctly. You may have to make changes to your Modelfile and run the `ollama create` command again in order to get everything correct.

After it is working, if you have created an account on [ollama.com](https://ollama.com), you can push your model to [ollama.com](https://ollama.com) to share it with other people. Use the name `<user name>/<model name>` (e.g. `jmorganca/my-model`) when you are creating the model, or use the `ollama cp` command to name the model with your user name at the beginning of the model name. You can then push the model using the command:

```bash
$ ollama push <user name>/<model name>
```

## Importing a GGUF based model

If you have a GGUF based model it is possible to import it into Ollama. You can obtain a GGUF model by:

  * Converting a Safetensors model with the `convert_hf_to_gguf.py` from Llama.cpp; or
  * Downloading a model from a place such as HuggingFace

To import the GGUF file, create a Modelfile:

```dockerfile
FROM /path/to/file.gguf
```

Not all models in GGUF format will work with Ollama.


## Quantizing a Model

Quantizing a model allows you to run models faster and with less memory consumption but at reduced accuracy. This allows you to run a model on more modest hardware.

Ollama can quantize FP16 and FP32 based models into different quantization levels using the `-q/--quantize` flag with the `ollama create` command.

### Supported Quantizations

- `Q4_0`
- `Q4_1`
- `Q5_0`
- `Q5_1`
- `Q8_0`

#### K-means Quantizations

- `Q3_K_S`
- `Q3_K_M`
- `Q3_K_L`
- `Q4_K_S`
- `Q4_K_M`
- `Q5_K_S`
- `Q5_K_M`
- `Q6_K`

First, create a Modelfile with the FP16 or FP32 based model you wish to quantize.

```dockerfile
FROM /path/to/my/gemma/f16/model
```

Use `ollama create` to then create the quantized model.

```shell
$ ollama create -q Q4_K_M mymodel
transferring model data
quantizing F16 model to Q4_K_M
creating new layer sha256:735e246cc1abfd06e9cdcf95504d6789a6cd1ad7577108a70d9902fef503c1bd
creating new layer sha256:0853f0ad24e5865173bbf9ffcc7b0f5d56b66fd690ab1009867e45e7d2c4db0f
writing manifest
success
```

