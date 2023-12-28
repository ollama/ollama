# Import a model

This guide walks through importing a GGUF, PyTorch or Safetensors model.

## Importing (GGUF)

### Step 1: Write a `Modelfile`

Start by creating a `Modelfile`. This file is the blueprint for your model, specifying weights, parameters, prompt templates and more.

```
FROM ./mistral-7b-v0.1.Q4_0.gguf
```

(Optional) many chat models require a prompt template in order to answer correctly. A default prompt template can be specified with the `TEMPLATE` instruction in the `Modelfile`:

```
FROM ./q4_0.bin
TEMPLATE "[INST] {{ .Prompt }} [/INST]"
```

### Step 2: Create the Ollama model

Finally, create a model from your `Modelfile`:

```
ollama create example -f Modelfile
```

### Step 3: Run your model

Next, test the model with `ollama run`:

```
ollama run example "What is your favourite condiment?"
```

## Importing (PyTorch & Safetensors)

### Supported models

Ollama supports a set of model architectures, with support for more coming soon:

- Llama & Mistral
- Falcon & RW
- BigCode

To view a model's architecture, check the `config.json` file in its HuggingFace repo. You should see an entry under `architectures` (e.g. `LlamaForCausalLM`).

### Step 1: Clone the HuggingFace repository (optional)

If the model is currently hosted in a HuggingFace repository, first clone that repository to download the raw model.

```
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
cd Mistral-7B-Instruct-v0.1
```

### Step 2: Convert and quantize to a `.bin` file (optional, for PyTorch and Safetensors)

If the model is in PyTorch or Safetensors format, a [Docker image](https://hub.docker.com/r/ollama/quantize) with the tooling required to convert and quantize models is available.

First, Install [Docker](https://www.docker.com/get-started/).

Next, to convert and quantize your model, run:

```
docker run --rm -v .:/model ollama/quantize -q q4_0 /model
```

This will output two files into the directory:

- `f16.bin`: the model converted to GGUF
- `q4_0.bin` the model quantized to a 4-bit quantization (Ollama will use this file to create the Ollama model)

### Step 3: Write a `Modelfile`

Next, create a `Modelfile` for your model:

```
FROM ./q4_0.bin
```

(Optional) many chat models require a prompt template in order to answer correctly. A default prompt template can be specified with the `TEMPLATE` instruction in the `Modelfile`:

```
FROM ./q4_0.bin
TEMPLATE "[INST] {{ .Prompt }} [/INST]"
```

### Step 4: Create the Ollama model

Finally, create a model from your `Modelfile`:

```
ollama create example -f Modelfile
```

### Step 5: Run your model

Next, test the model with `ollama run`:

```
ollama run example "What is your favourite condiment?"
```

## Publishing your model (optional – early alpha)

Publishing models is in early alpha. If you'd like to publish your model to share with others, follow these steps:

1. Create [an account](https://ollama.ai/signup)
2. Run `cat ~/.ollama/id_ed25519.pub` to view your Ollama public key. Copy this to the clipboard.
3. Add your public key to your [Ollama account](https://ollama.ai/settings/keys)

Next, copy your model to your username's namespace:

```
ollama cp example <your username>/example
```

Then push the model:

```
ollama push <your username>/example
```

After publishing, your model will be available at `https://ollama.ai/<your username>/example`.

## Quantization reference

The quantization options are as follow (from highest highest to lowest levels of quantization). Note: some architectures such as Falcon do not support K quants.

- `q2_K`
- `q3_K`
- `q3_K_S`
- `q3_K_M`
- `q3_K_L`
- `q4_0` (recommended)
- `q4_1`
- `q4_K`
- `q4_K_S`
- `q4_K_M`
- `q5_0`
- `q5_1`
- `q5_K`
- `q5_K_S`
- `q5_K_M`
- `q6_K`
- `q8_0`

## Manually converting & quantizing models

### Prerequisites

Start by cloning the `llama.cpp` repo to your machine in another directory:

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

Next, install the Python dependencies:

```
pip install -r requirements.txt
```

Finally, build the `quantize` tool:

```
make quantize
```

### Convert the model

Run the correct conversion script for your model architecture:

```shell
# LlamaForCausalLM or MistralForCausalLM
python convert.py <path to model directory>

# FalconForCausalLM
python convert-falcon-hf-to-gguf.py <path to model directory>

# GPTBigCodeForCausalLM
python convert-starcoder-hf-to-gguf.py <path to model directory>
```

### Quantize the model

```
quantize <path to model dir>/ggml-model-f32.bin <path to model dir>/q4_0.bin q4_0
```
