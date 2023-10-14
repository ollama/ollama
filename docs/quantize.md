# How to Quantize a Model

Sometimes the model you want to work with is not available at [https://ollama.ai/library](https://ollama.ai/library). If you want to try out that model before we have a chance to quantize it, you can use this process.

## Figure out if we can run the model?

Not all models will work with Ollama. There are a number of factors that go into whether we are able to work with the next cool model. First it has to work with llama.cpp. Then we have to have implemented the features of llama.cpp that it requires. And then, sometimes, even with both of those, the model might not work...

1. What is the model you want to convert and upload?
2. Visit the model's page on HuggingFace.
3. Switch to the **Files and versions** tab.
4. Click on the **config.json** file. If there is no config.json file, it may not work.
5. Take note of the **architecture** list in the json file.
6. Does any entry in the list match one of the following architectures?
    1. LlamaForCausalLM
    2. MistralForCausalLM
    3. RWForCausalLM
    4. FalconForCausalLM
    5. GPTNeoXForCausalLM
    6. GPTBigCodeForCausalLM
7. If the answer is yes, then there is a good chance the model will run after being converted and quantized.
8. An alternative to this process is to visit [https://caniquant.tvl.st](https://caniquant.tvl.st) and enter the org/modelname in the box and submit.

At this point there are two processes you can use. You can either use a Docker container to convert and quantize, OR you can manually run the scripts. The Docker container is the easiest way to do it, but it requires you to have Docker installed on your machine. If you don't have Docker installed, you can follow the manual process.

## Convert and Quantize with Docker

Run `docker run --rm -v /path/to/model/repo:/repo ollama/quantize -q quantlevel /repo`. For instance, if you have downloaded the latest Mistral 7B model, then clone it to your machine. Then change into that directory and you can run:

```shell
docker run --rm -v .:/repo ollama/quantize -q q4_0 /repo
```

You can find the different quantization levels below under **Quantize the Model**.

This will output two files into the directory. First is a f16.bin file that is the model converted to GGUF. The second file is a q4_0.bin file which is the model quantized to a 4 bit quantization. You should rename it to something more descriptive.

You can find the repository for the Docker container here: [https://github.com/mxyng/quantize](https://github.com/mxyng/quantize)

## Convert and Quantize Manually

### Clone llama.cpp to your machine

If we know the model has a chance of working, then we need to convert and quantize. This is a matter of running two separate scripts in the llama.cpp project.

1. Decide where you want the llama.cpp repository on your machine.
2. Navigate to that location and then run:
 [`git clone https://github.com/ggerganov/llama.cpp.git`](https://github.com/ggerganov/llama.cpp.git)
    1. If you don't have git installed, download this zip file and unzip it to that location: https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip
3. Install the Python dependencies: `pip install torch transformers sentencepiece`

### Convert the model to GGUF

1. Decide on the right convert script to run. What was the model architecture you found in the first section.
    1. LlamaForCausalLM or MistralForCausalLM:
    run `python3 convert.py <modelfilename>`
    No need to specify fp16 or fp32.
    2. FalconForCausalLM or RWForCausalLM:
    run `python3 convert-falcon-hf-to-gguf.py <modelfilename> <fpsize>`  
    fpsize depends on the weight size. 1 for fp16, 0 for fp32
    3. GPTNeoXForCausalLM:
    run `python3 convert-gptneox-hf-to-gguf.py <modelfilename> <fpsize>`
    fpsize depends on the weight size. 1 for fp16, 0 for fp32
    4. GPTBigCodeForCausalLM:
    run `python3 convert-starcoder-hf-to-gguf.py <modelfilename> <fpsize>`
    fpsize depends on the weight size. 1 for fp16, 0 for fp32

### Quantize the model

If the model converted successfully, there is a good chance it will also quantize successfully. Now you need to decide on the quantization to use. We will always try to create all the quantizations and upload them to the library. You should decide which level is more important to you and quantize accordingly.

The quantization options are as follows. Note that some architectures such as Falcon do not support K quants.

- Q4_0
- Q4_1
- Q5_0
- Q5_1
- Q2_K
- Q3_K
- Q3_K_S
- Q3_K_M
- Q3_K_L
- Q4_K
- Q4_K_S
- Q4_K_M
- Q5_K
- Q5_K_S
- Q5_K_M
- Q6_K
- Q8_0

Run the following command `quantize <converted model from above> <output file> <quantization type>`

## Now Create the Model

Now you can create the Ollama model. Refer to the [modelfile](./modelfile.md) doc for more information on doing that.
