# Unsloth FastLanguageModel to ollama

This guide provides information on how to set the fine-tuned model we trained using unsloth from a Google Colab training notebook and call the model locally via the Ollama cli.

## Problem Statement

Most people do not own expensive GPUs but can afford to spend $9.99 on Google Colab Pro. The problem is that after training, we still have to evaluate our model’s output, which means more GPU runtime credits are spent. Thus, running our inference locally and exclusively spending credits on training will give us a ton of savings. In addition, Unsloth provides even more savings on training costs.

## Prerequisites

To successfully run the fine-tuned model, we need:

1. Huggingface account
2. A Base unsloth model - for this guide, we have chosen `unsloth/tinyllama` as the base model
3. A basic understanding of the unsloth FastLanguageModel. In particular, fine-tuning unsloth/tinyllama. We recommend their Google Colab training notebooks on huggingface for more information on the training data
   - https://huggingface.co/unsloth/tinyllama
4. The Lora adapters that were saved online via the huggingface hub
5. A working local ollama installation: as of writing, we used 0.1.32, but it should work from later versions.
   - `ollama --version`
   - `ollama version is 0.1.32`

## Training

To recall, we provided some training code using unsloth FastLanguageModel. Please note that we can log in on huggingface on Google Colab by setting our API token as a secret token labeled “HF_TOKEN”

```
import os
from google.colab import userdata
hf_token = userdata.get("HF_TOKEN")
os.environ['HF_TOKEN'] = hf_token
```

We then run the cli command below to login

```
!huggingface-cli login --token $HF_TOKEN
```

To check our token is working, run

```
!huggingface-cli whoami
```

Below is a sample training code from the Unsloth notebook

```
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/tinyllama", # "unsloth/tinyllama" for 16bit loading
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

Moreover, we used the training code below. We provided `dataset` and `eval_dataset` for our training data, which had only one `text` column.

```
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.utils import logging
logging.set_verbosity_info()

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True, # Packs short sequences together to save time!
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 2,
        learning_rate = 2e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.1,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
```

Then, we should be able to run our inference, as shown below.

```
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
"""
<s>
Q:
What is the capital of France?
A:
"""
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 1000, use_cache = True)

print(tokenizer.batch_decode(outputs))
```

Lastly, below, we demonstrate how to save the model online via huggingface

```
model.push_to_hub_merged(“myhfusername/my-model", tokenizer, save_method = "lora")
```

## Installation

When we wrote part of this guide we merely took from the page below
https://rentry.org/llama-cpp-conversions#setup

### 1. Build llama.cpp

Clone the llama.cpp repository using

```
git clone https://github.com/ggerganov/llama.cpp
```

```
cd llama.cpp
```

`llama.cpp` has Python scripts that we need to run, so we need to `pip install` its dependencies

`pip install -r requirements.txt`

Now, let us build our local llama.cpp

`make clean && make all -j`

For anyone with nvidia GPUs
`make clean && LLAMA_CUDA=1 make all -j`

### 2. Clone our huggingface base model and the Lora adapters from huggingface hub we uploaded earlier, where we used the `push_to_hub_merged()` function

From the llama.cpp folder let us clone our base model.

```
git clone https://huggingface.co/unsloth/tinyllama
```

Next, we clone our Lora model

```
git clone https://huggingface.co/myhfusername/my-model
```

### 3. GGUF conversion

We now need to convert both the base model and the Lora adapters. Note that the base model has a pre-quantization step before the conversion

```
python convert.py tinyllama --outtype f16 --outfile tinyllama.f16.gguf
```

### 4. GGUF conversion of Lora adapters

```
python convert-lora-to-ggml.py my-model
```

If the conversion succeeds, the last lines from our output should be

```
Converted my-model/adapter_config.json and my-model/adapter_model.safetensors to my-model/ggml-adapter-model.bin
```

### 5. Merge our gguf base model and adapter model using the command`export-lora`

--model-base - is the gguf model
--model-out - is the new gguf model
--lora is the adapter model

```
export-lora --model-base tinyllama.f16.gguf --model-out tinyllama-my-model.gguf --lora my-model/ggml-adapter-model.bin
```

## 6. Create ollama Modelfile

```
FROM tinyllama-my-model.gguf

# set the system message
SYSTEM """
You are a super helpful helper.
"""

PARAMETER stop <s>
PARAMETER stop </s>
```

## 7. Create a Modelfile

`ollama create my-model -f Modelfile`

## 8. Test command

`ollama run my-model "<s>\nQ: \nWhat is the capital of France?\nA:\n"`

Author: Jed Tiotuico

Github: http://github.com/jedt
