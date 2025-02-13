# FAQ

## How can I upgrade Ollama?

Ollama on macOS and Windows will automatically download updates. Click on the taskbar or menubar item and then click "Restart to update" to apply the update. Updates can also be installed by downloading the latest version [manually](https://ollama.com/download/).

On Linux, re-run the install script:

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

## How can I view the logs?

Review the [Troubleshooting](./troubleshooting.md) docs for more about using logs.

## Is my GPU compatible with Ollama?

Please refer to the [GPU docs](./gpu.md).

## How can I specify the context window size?

By default, Ollama uses a context window size of 2048 tokens.

To change this when using `ollama run`, use `/set parameter`:

```
/set parameter num_ctx 4096
```

When using the API, specify the `num_ctx` parameter:

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?",
  "options": {
    "num_ctx": 4096
  }
}'
```

## How can I tell if my model was loaded onto the GPU?

Use the `ollama ps` command to see what models are currently loaded into memory.

```shell
ollama ps
NAME      	ID          	SIZE 	PROCESSOR	UNTIL
llama3:70b	bcfb190ca3a7	42 GB	100% GPU 	4 minutes from now
```

The `Processor` column will show which memory the model was loaded in to:
* `100% GPU` means the model was loaded entirely into the GPU
* `100% CPU` means the model was loaded entirely in system memory
* `48%/52% CPU/GPU` means the model was loaded partially onto both the GPU and into system memory

## How do I configure Ollama server?

Ollama server can be configured with environment variables.

### Setting environment variables on Mac

If Ollama is run as a macOS application, environment variables should be set using `launchctl`:

1. For each environment variable, call `launchctl setenv`.

    ```bash
    launchctl setenv OLLAMA_HOST "0.0.0.0"
    ```

2. Restart Ollama application.

### Setting environment variables on Linux

If Ollama is run as a systemd service, environment variables should be set using `systemctl`:

1. Edit the systemd service by calling `systemctl edit ollama.service`. This will open an editor.

2. For each environment variable, add a line `Environment` under section `[Service]`:

    ```ini
    [Service]
    Environment="OLLAMA_HOST=0.0.0.0"
    ```

3. Save and exit.

4. Reload `systemd` and restart Ollama:

   ```bash
   systemctl daemon-reload
   systemctl restart ollama
   ```

### Setting environment variables on Windows

On Windows, Ollama inherits your user and system environment variables.

1. First Quit Ollama by clicking on it in the task bar.

2. Start the Settings (Windows 11) or Control Panel (Windows 10) application and search for _environment variables_.

3. Click on _Edit environment variables for your account_.

4. Edit or create a new variable for your user account for `OLLAMA_HOST`, `OLLAMA_MODELS`, etc.

5. Click OK/Apply to save.

6. Start the Ollama application from the Windows Start menu.

## How do I use Ollama behind a proxy?

Ollama pulls models from the Internet and may require a proxy server to access the models. Use `HTTPS_PROXY` to redirect outbound requests through the proxy. Ensure the proxy certificate is installed as a system certificate. Refer to the section above for how to use environment variables on your platform.

> [!NOTE]
> Avoid setting `HTTP_PROXY`. Ollama does not use HTTP for model pulls, only HTTPS. Setting `HTTP_PROXY` may interrupt client connections to the server.

### How do I use Ollama behind a proxy in Docker?

The Ollama Docker container image can be configured to use a proxy by passing `-e HTTPS_PROXY=https://proxy.example.com` when starting the container.

Alternatively, the Docker daemon can be configured to use a proxy. Instructions are available for Docker Desktop on [macOS](https://docs.docker.com/desktop/settings/mac/#proxies), [Windows](https://docs.docker.com/desktop/settings/windows/#proxies), and [Linux](https://docs.docker.com/desktop/settings/linux/#proxies), and Docker [daemon with systemd](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy).

Ensure the certificate is installed as a system certificate when using HTTPS. This may require a new Docker image when using a self-signed certificate.

```dockerfile
FROM ollama/ollama
COPY my-ca.pem /usr/local/share/ca-certificates/my-ca.crt
RUN update-ca-certificates
```

Build and run this image:

```shell
docker build -t ollama-with-ca .
docker run -d -e HTTPS_PROXY=https://my.proxy.example.com -p 11434:11434 ollama-with-ca
```

## Does Ollama send my prompts and answers back to ollama.com?

No. Ollama runs locally, and conversation data does not leave your machine.

## How can I expose Ollama on my network?

Ollama binds 127.0.0.1 port 11434 by default. Change the bind address with the `OLLAMA_HOST` environment variable.

Refer to the section [above](#how-do-i-configure-ollama-server) for how to set environment variables on your platform.

## How can I use Ollama with a proxy server?

Ollama runs an HTTP server and can be exposed using a proxy server such as Nginx. To do so, configure the proxy to forward requests and optionally set required headers (if not exposing Ollama on the network). For example, with Nginx:

```nginx
server {
    listen 80;
    server_name example.com;  # Replace with your domain or IP
    location / {
        proxy_pass http://localhost:11434;
        proxy_set_header Host localhost:11434;
    }
}
```

## How can I use Ollama with ngrok?

Ollama can be accessed using a range of tools for tunneling tools. For example with Ngrok:

```shell
ngrok http 11434 --host-header="localhost:11434"
```

## How can I use Ollama with Cloudflare Tunnel?

To use Ollama with Cloudflare Tunnel, use the `--url` and `--http-host-header` flags:

```shell
cloudflared tunnel --url http://localhost:11434 --http-host-header="localhost:11434"
```

## How can I allow additional web origins to access Ollama?

Ollama allows cross-origin requests from `127.0.0.1` and `0.0.0.0` by default. Additional origins can be configured with `OLLAMA_ORIGINS`.

Refer to the section [above](#how-do-i-configure-ollama-server) for how to set environment variables on your platform.

## Where are models stored?

- macOS: `~/.ollama/models`
- Linux: `/usr/share/ollama/.ollama/models`
- Windows: `C:\Users\%username%\.ollama\models`

### How do I set them to a different location?

If a different directory needs to be used, set the environment variable `OLLAMA_MODELS` to the chosen directory.

> Note: on Linux using the standard installer, the `ollama` user needs read and write access to the specified directory. To assign the directory to the `ollama` user run `sudo chown -R ollama:ollama <directory>`.

Refer to the section [above](#how-do-i-configure-ollama-server) for how to set environment variables on your platform.

## How can I use Ollama in Visual Studio Code?

There is already a large collection of plugins available for VSCode as well as other editors that leverage Ollama. See the list of [extensions & plugins](https://github.com/ollama/ollama#extensions--plugins) at the bottom of the main repository readme.

## How do I use Ollama with GPU acceleration in Docker?

The Ollama Docker container can be configured with GPU acceleration in Linux or Windows (with WSL2). This requires the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). See [ollama/ollama](https://hub.docker.com/r/ollama/ollama) for more details.

GPU acceleration is not available for Docker Desktop in macOS due to the lack of GPU passthrough and emulation.

## Why is networking slow in WSL2 on Windows 10?

This can impact both installing Ollama, as well as downloading models.

Open `Control Panel > Networking and Internet > View network status and tasks` and click on `Change adapter settings` on the left panel. Find the `vEthernel (WSL)` adapter, right click and select `Properties`.
Click on `Configure` and open the `Advanced` tab. Search through each of the properties until you find `Large Send Offload Version 2 (IPv4)` and `Large Send Offload Version 2 (IPv6)`. *Disable* both of these
properties.

## How can I preload a model into Ollama to get faster response times?

If you are using the API you can preload a model by sending the Ollama server an empty request. This works with both the `/api/generate` and `/api/chat` API endpoints.

To preload the mistral model using the generate endpoint, use:
```shell
curl http://localhost:11434/api/generate -d '{"model": "mistral"}'
```

To use the chat completions endpoint, use:
```shell
curl http://localhost:11434/api/chat -d '{"model": "mistral"}'
```

To preload a model using the CLI, use the command:
```shell
ollama run llama3.2 ""
```

## How do I keep a model loaded in memory or make it unload immediately?

By default models are kept in memory for 5 minutes before being unloaded. This allows for quicker response times if you're making numerous requests to the LLM. If you want to immediately unload a model from memory, use the `ollama stop` command:

```shell
ollama stop llama3.2
```

If you're using the API, use the `keep_alive` parameter with the `/api/generate` and `/api/chat` endpoints to set the amount of time that a model stays in memory. The `keep_alive` parameter can be set to:
* a duration string (such as "10m" or "24h")
* a number in seconds (such as 3600)
* any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
* '0' which will unload the model immediately after generating a response

For example, to preload a model and leave it in memory use:
```shell
curl http://localhost:11434/api/generate -d '{"model": "llama3.2", "keep_alive": -1}'
```

To unload the model and free up memory use:
```shell
curl http://localhost:11434/api/generate -d '{"model": "llama3.2", "keep_alive": 0}'
```

Alternatively, you can change the amount of time all models are loaded into memory by setting the `OLLAMA_KEEP_ALIVE` environment variable when starting the Ollama server. The `OLLAMA_KEEP_ALIVE` variable uses the same parameter types as the `keep_alive` parameter types mentioned above. Refer to the section explaining [how to configure the Ollama server](#how-do-i-configure-ollama-server) to correctly set the environment variable.

The `keep_alive` API parameter with the `/api/generate` and `/api/chat` API endpoints will override the `OLLAMA_KEEP_ALIVE` setting.

## How do I manage the maximum number of requests the Ollama server can queue?

If too many requests are sent to the server, it will respond with a 503 error indicating the server is overloaded.  You can adjust how many requests may be queue by setting `OLLAMA_MAX_QUEUE`.

## How does Ollama handle concurrent requests?

Ollama supports two levels of concurrent processing.  If your system has sufficient available memory (system memory when using CPU inference, or VRAM for GPU inference) then multiple models can be loaded at the same time.  For a given model, if there is sufficient available memory when the model is loaded, it is configured to allow parallel request processing.

If there is insufficient available memory to load a new model request while one or more models are already loaded, all new requests will be queued until the new model can be loaded.  As prior models become idle, one or more will be unloaded to make room for the new model.  Queued requests will be processed in order.  When using GPU inference new models must be able to completely fit in VRAM to allow concurrent model loads.

Parallel request processing for a given model results in increasing the context size by the number of parallel requests.  For example, a 2K context with 4 parallel requests will result in an 8K context and additional memory allocation.

The following server settings may be used to adjust how Ollama handles concurrent requests on most platforms:

- `OLLAMA_MAX_LOADED_MODELS` - The maximum number of models that can be loaded concurrently provided they fit in available memory.  The default is 3 * the number of GPUs or 3 for CPU inference.
- `OLLAMA_NUM_PARALLEL` - The maximum number of parallel requests each model will process at the same time.  The default will auto-select either 4 or 1 based on available memory.
- `OLLAMA_MAX_QUEUE` - The maximum number of requests Ollama will queue when busy before rejecting additional requests. The default is 512

Note: Windows with Radeon GPUs currently default to 1 model maximum due to limitations in ROCm v5.7 for available VRAM reporting.  Once ROCm v6.2 is available, Windows Radeon will follow the defaults above.  You may enable concurrent model loads on Radeon on Windows, but ensure you don't load more models than will fit into your GPUs VRAM.

## How does Ollama load models on multiple GPUs?

When loading a new model, Ollama evaluates the required VRAM for the model against what is currently available.  If the model will entirely fit on any single GPU, Ollama will load the model on that GPU.  This typically provides the best performance as it reduces the amount of data transferring across the PCI bus during inference.  If the model does not fit entirely on one GPU, then it will be spread across all the available GPUs.

## How can I enable Flash Attention?

Flash Attention is a feature of most modern models that can significantly reduce memory usage as the context size grows.  To enable Flash Attention, set the `OLLAMA_FLASH_ATTENTION` environment variable to `1` when starting the Ollama server.

## How can I set the quantization type for the K/V cache?

The K/V context cache can be quantized to significantly reduce memory usage when Flash Attention is enabled.

To use quantized K/V cache with Ollama you can set the following environment variable:

- `OLLAMA_KV_CACHE_TYPE` - The quantization type for the K/V cache.  Default is `f16`.

> Note: Currently this is a global option - meaning all models will run with the specified quantization type.

The currently available K/V cache quantization types are:

- `f16` - high precision and memory usage (default).
- `q8_0` - 8-bit quantization, uses approximately 1/2 the memory of `f16` with a very small loss in precision, this usually has no noticeable impact on the model's quality (recommended if not using f16).
- `q4_0` - 4-bit quantization, uses approximately 1/4 the memory of `f16` with a small-medium loss in precision that may be more noticeable at higher context sizes.

How much the cache quantization impacts the model's response quality will depend on the model and the task.  Models that have a high GQA count (e.g. Qwen2) may see a larger impact on precision from quantization than models with a low GQA count.

You may need to experiment with different quantization types to find the best balance between memory usage and quality.

## How can I prevent OOMs?

Memory calculations depend on the architecture of the model and sometimes it's not correct.  This can lead to the runner terminating from an OOM.  This may be mitigated by one or more of the following:

1. Set [`OLLAMA_GPU_OVERHEAD`](https://github.com/ollama/ollama/blob/5f8051180e3b9aeafc153f6b5056e7358a939c88/envconfig/config.go#L237) to give llama.cpp a buffer to grow in to (eg, `OLLAMA_GPU_OVERHEAD=536870912` to reserve 512M).  The exact value depends on model/GPU.
2. Enable flash attention by setting [`OLLAMA_FLASH_ATTENTION=1`](https://github.com/ollama/ollama/blob/5f8051180e3b9aeafc153f6b5056e7358a939c88/envconfig/config.go#L236) in the server environment.  Flash attention is a more efficient use of memory and may reduce memory pressure.  See above.
3. Reduce the number layers that ollama thinks it can offload to the GPU, see [here](https://github.com/ollama/ollama/issues/6950#issuecomment-2373663650).
4. In Linux, set `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`.  This will allow the GPU to offload to CPU memory if VRAM is exhausted.  This is only useful for small amounts of memory as there is a [performance penalty](https://github.com/ollama/ollama/issues/7584#issuecomment-2466715900).  However, in the case where the goal is to reduce OOMs, the amount offloaded will be small and the impact minimal. 
5. Set [`OLLAMA_NUM_PARALLEL`](https://github.com/ollama/ollama/blob/a4f69a0191b304c204ef074ccd6523f121bfddfe/envconfig/config.go#L249) to 1.
6. Using a smaller context buffer by reducing [`num_ctx`](https://github.com/ollama/ollama/blob/a4f69a0191b304c204ef074ccd6523f121bfddfe/docs/modelfile.md#valid-parameters-and-values:~:text=mirostat_tau%205.0-,num_ctx,-Sets%20the%20size).

## How can I set the size of the context buffer?

The size can be set in an API call (`"options":{"num_ctx":xxx}}`), in the ollama CLI (`/set parameter num_ctx xxx`), or configured in the model.  When using the OpenAI API compatability endpoint, setting the context size is not currently supported so it must be configured in the model.

There are two ways to configure the context size in a model.  First, use the `/save` command in the ollama CLI:
```console
ollama run model-with-normal-context
>>> /set parameter num_ctx 4096
>>> /save model-with-4096-context
>>> /bye
```
Second, create a new Modelfile:
```console
ollama show --modelfile model-with-normal-context > Modelfile
echo PARAMETER num_ctx 4096 > Modelfile
ollama create model-with-4096-context
```
After the model has been created, use it in the normal way:
```console
curl localhost:11434/api/generate -d "{\"model\":\"model-with-4096-context\",\"prompt\":\"why is the sky blue?\"}"
```
```console
ollama run model-with-4096-context
```
If the model is large, the second approach above can cause a lot of disk activity as ollama loads the GGUF file.  This can be prevented by editing the Modelfile, uncommenting the first `FROM` statement and commenting the second `FROM` statement.

## Why does the runner use 4 times the context buffer I configured?

Ollama can do parallel completions, controlled by `OLLAMA_NUM_PARALLEL`.  If unset, the default is 4 (or 1 if your system has only a small amount of free RAM).  Each completion instance has its own context buffer, size given by `num_ctx`.  The value of `ctx-size` on the runner command line or the value `n_ctx` in the logs is the total context buffer that is allocated, `OLLAMA_NUM_PARALLEL` * `num_ctx`.

## What is K-shift and why did it kill ollama?

When the context buffer fills up during completion, the inference engine wants to shift the buffer to make room for new tokens.  Some model architectures, notably Deepseek, do not support this feature and the runner will exit with an error.  To prevent this, the context buffer (`num_ctx`) needs to be sized such that it can contain both input tokens and output tokens.  To prevent the completion from generating more output tokens than expected, `num_predict` should be set.  For example, if the model is expected to received up to 10,000 tokens and generate no more than 5,000 tokens in response, `num_ctx` is set to 15,000 and `num_predict` to 5,000.  These parameters can be set in the `options` field of an API call, or configured directly in the model.

For example, to create a copy of deepseek-r1:7b that can accept 10,000 input tokens:
```sh
echo FROM deepseek-r1:7b > Modelfile
echo PARAMETER num_ctx 15000 >> Modelfile
echo PARAMETER num_predict 5000 >> Modelfile
ollama create deepseek-r1:7b-10kcontext
```

## How can I force a model to load only in RAM?

Set `num_gpu` to zero.  This can be done in the `options` field of an API call, in the ollama CLI (`/set parameter num_gpu xxx`), or configured directly in the model.
```sh
echo FROM deepseek-r1:7b > Modelfile
echo PARAMETER num_gpu 0 >> Modelfile
ollama create deepseek-r1:7b-cpu
```

## How can I force a model to load only in VRAM?

ollama preferentially loads models in to VRAM, only loading part of a model in system RAM if it calculates the the whole model will not fit in VRAM.  This can be overriden by setting `num_gpu`.
```sh
echo FROM deepseek-r1:7b > Modelfile
echo PARAMETER num_gpu 999 >> Modelfile
ollama create deepseek-r1:7b-cpu
```
Warning: this can lead to runner crashes or performance penalties.

## What's the impact of using shared memory?

Nvidia GPUs have shared memory, or the ability of the GPU to access system RAM.  This makes it technically feasible for the GPU to do all inference processing even if part of the model is residing in system RAM.  However, there is a performance penalty.  If more than a few layers are loaded in system RAM and processed by the GPU, token generation rate can be affected.

## Does using multuple GPUs increase token generation rate?

A model is a collection of layers.  The layers are processed sequentially and processing in a layer must be compete before the output of the layer can be fed into the next layer.  Consequently, multiple GPUs do not increase the token generation rate for an individual completion.

<!--
## I'm unable to connect to the ollama server in Windows.

## How can I make a model access the internet?

## I'm having problems downloading a model.

## Why is my GPU not being used?
-->
