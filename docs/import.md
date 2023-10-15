# Import a model

This guide walks through creating an Ollama model from an existing model on HuggingFace from PyTorch, Safetensors or GGUF. It optionally covers pushing the model to [ollama.ai](https://ollama.ai/library).

## Supported models

Ollama supports a set of model architectures, with support for more coming soon:

- Llama
- Mistral
- Falcon & RW
- GPTNeoX
- BigCode

To view a model's architecture, check its `config.json` file. You should see an entry under `architecture` (e.g. `LlamaForCausalLM`).

## Importing

### Step 1: Clone the HuggingFace repository

```
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
cd Mistral-7B-Instruct-v0.1
```

### Step 2: Convert and quantize

- Install [Docker](https://www.docker.com/get-started/)

Until Ollama supports conversion and quantization as a built-in feature, a [Docker image](https://hub.docker.com/r/ollama/quantize) with this tooling built-in is available.

To convert and quantize your model, run:

```
docker run --rm -v .:/model ollama/quantize -q q4_0 /model
```

This will output two files into the directory:

- `f16.bin`: the model converted to GGUF
- `q4_0.bin` the model quantized to a 4-bit quantization

### Step 3: Write a `Modelfile`

Next, create a `Modelfile` for your model. This file is the blueprint for your model, specifying weights, parameters, prompt templates and more.

```
FROM ./q4_0.bin
```

(Optional) many chat models require a prompt template in order to answer correctly. A default prompt template can be specified with the `TEMPLATE` instruction in the `Modelfile`:

```
FROM ./q4_0.bin
TEMPLATE "[INST] {{ .Prompt }} [/INST]"
```

### Step 4: Create an Ollama model

Finally, create a model from your `Modelfile`:

```
ollama create example -f Modelfile
```

Next, test the model with `ollama run`:

```
ollama run example "What is your favourite condiment?"
```

### Step 5: Publish your model (optional - in alpha)

Publishing models is in early alpha. If you'd like to publish your model to share with others, follow these steps:

1. Create [an account](https://ollama.ai/signup)
2. Ollama uses SSH keys similar to Git. Find your public key with `cat ~/.ollama/id_ed25519.pub` and copy it to your clipboard.
3. Add your public key to your [Ollama account](https://ollama.ai/settings/keys)

Next, copy your model to your username's namespace:

```
ollama cp example <your username>/example
```

Then push the model:

```
ollama push <your username>/example
```

After publishing, your model will be available at `https://ollama.ai/<your username>/example`

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
python3 convert.py <path to model directory>

# FalconForCausalLM
python3 convert-falcon-hf-to-gguf.py <path to model directory>

# GPTNeoXForCausalLM
python3 convert-falcon-hf-to-gguf.py <path to model directory>

# GPTBigCodeForCausalLM
python3 convert-starcoder-hf-to-gguf.py <path to model directory>
```

### Quantize the model

```
quantize <path to model dir>/ggml-model-f32.bin <path to model dir>/q4_0.bin q4_0
```
