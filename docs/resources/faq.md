# Frequently Asked Questions

This page provides answers to common questions about Ollama. The questions are organized by category for easier navigation.

## Installation and Updates

### How can I upgrade Ollama?

Ollama on macOS and Windows will automatically download updates. Click on the taskbar or menubar item and then click "Restart to update" to apply the update. Updates can also be installed by downloading the latest version [manually](https://ollama.com/download/).

On Linux, re-run the install script:

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

### How can I install older or pre-release versions on Linux?

If you run into problems on Linux and want to install an older version, or you'd like to try out a pre-release before it's officially released, you can tell the install script which version to install:

```shell
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.5.7 sh
```

## Configuration

### How do I configure Ollama server?

Ollama server can be configured with environment variables. The method for setting these variables depends on your platform:

#### macOS

If Ollama is run as a macOS application, environment variables should be set using `launchctl`:

1. For each environment variable, call `launchctl setenv`.

    ```bash
    launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
    ```

2. Restart Ollama application.

#### Linux

If Ollama is run as a systemd service, environment variables should be set using `systemctl`:

1. Edit the systemd service by calling `systemctl edit ollama.service`. This will open an editor.

2. For each environment variable, add a line `Environment` under section `[Service]`:

    ```ini
    [Service]
    Environment="OLLAMA_HOST=0.0.0.0:11434"
    ```

3. Save and exit.

4. Reload `systemd` and restart Ollama:

   ```shell
   systemctl daemon-reload
   systemctl restart ollama
   ```

#### Windows

On Windows, Ollama inherits your user and system environment variables.

1. First Quit Ollama by clicking on it in the task bar.

2. Start the Settings (Windows 11) or Control Panel (Windows 10) application and search for _environment variables_.

3. Click on _Edit environment variables for your account_.

4. Edit or create a new variable for your user account for `OLLAMA_HOST`, `OLLAMA_MODELS`, etc.

5. Click OK/Apply to save.

6. Start the Ollama application from the Windows Start menu.

### Where are models stored?

- macOS: `~/.ollama/models`
- Linux: `/usr/share/ollama/.ollama/models`
- Windows: `C:\Users\%username%\.ollama\models`

### How do I set models to a different location?

If a different directory needs to be used, set the environment variable `OLLAMA_MODELS` to the chosen directory.

> Note: on Linux using the standard installer, the `ollama` user needs read and write access to the specified directory. To assign the directory to the `ollama` user run `sudo chown -R ollama:ollama <directory>`.

## Performance and Hardware

### Is my GPU compatible with Ollama?

Please refer to the [GPU documentation](gpu.md) for detailed information about GPU compatibility.

### How can I tell if my model was loaded onto the GPU?

Use the `ollama ps` command to see what models are currently loaded into memory.

```shell
ollama ps
```

> **Output**:
>
> ```
> NAME      	ID          	SIZE 	PROCESSOR	UNTIL
> llama3:70b	bcfb190ca3a7	42 GB	100% GPU 	4 minutes from now
> ```

The `Processor` column will show which memory the model was loaded in to:
* `100% GPU` means the model was loaded entirely into the GPU
* `100% CPU` means the model was loaded entirely in system memory
* `48%/52% CPU/GPU` means the model was loaded partially onto both the GPU and into system memory

### How can I specify the context window size?

By default, Ollama uses a context window size of 4096 tokens. 

This can be overridden with the `OLLAMA_CONTEXT_LENGTH` environment variable. For example, to set the default context window to 8K, use: 

```shell
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

To change this when using `ollama run`, use `/set parameter`:

```shell
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

### How can I preload a model into Ollama to get faster response times?

If you are using the API you can preload a model by sending the Ollama server an empty request. This works with both the `/api/generate` and `/api/chat` API endpoints.

To preload the mistral model using the generate endpoint, use:

```shell
curl http://localhost:11434/api/generate -d '{"model": "mistral"}'
```

To preload a model using the CLI, use the command:

```shell
ollama run llama3.2 ""
```

### How do I keep a model loaded in memory or make it unload immediately?

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

## Networking and Security

### How can I expose Ollama on my network?

Ollama binds 127.0.0.1 port 11434 by default. Change the bind address with the `OLLAMA_HOST` environment variable.

### How can I use Ollama with a proxy server?

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

### How do I use Ollama behind a proxy?

Ollama pulls models from the Internet and may require a proxy server to access the models. Use `HTTPS_PROXY` to redirect outbound requests through the proxy. Ensure the proxy certificate is installed as a system certificate.

> [!NOTE]
> Avoid setting `HTTP_PROXY`. Ollama does not use HTTP for model pulls, only HTTPS. Setting `HTTP_PROXY` may interrupt client connections to the server.

### How can I allow additional web origins to access Ollama?

Ollama allows cross-origin requests from `127.0.0.1` and `0.0.0.0` by default. Additional origins can be configured with `OLLAMA_ORIGINS`.

For browser extensions, you'll need to explicitly allow the extension's origin pattern. Set `OLLAMA_ORIGINS` to include `chrome-extension://*`, `moz-extension://*`, and `safari-web-extension://*` if you wish to allow all browser extensions access, or specific extensions as needed:

```
# Allow all Chrome, Firefox, and Safari extensions
OLLAMA_ORIGINS=chrome-extension://*,moz-extension://*,safari-web-extension://* ollama serve
```

### Does Ollama send my prompts and answers back to ollama.com?

No. Ollama runs locally, and conversation data does not leave your machine.

## Integration with Other Tools

### How can I use Ollama in Visual Studio Code?

There is already a large collection of plugins available for VSCode as well as other editors that leverage Ollama. See the list of [extensions & plugins](https://github.com/ollama/ollama#extensions--plugins) at the bottom of the main repository readme.

### How do I use Ollama with GPU acceleration in Docker?

The Ollama Docker container can be configured with GPU acceleration in Linux or Windows (with WSL2). This requires the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). See [ollama/ollama](https://hub.docker.com/r/ollama/ollama) for more details.

GPU acceleration is not available for Docker Desktop in macOS due to the lack of GPU passthrough and emulation.

## Troubleshooting

### How can I view the logs?

Review the [Troubleshooting](./troubleshooting.md) docs for more about using logs.

### Why is networking slow in WSL2 on Windows 10?

This can impact both installing Ollama, as well as downloading models.

Open `Control Panel > Networking and Internet > View network status and tasks` and click on `Change adapter settings` on the left panel. Find the `vEthernel (WSL)` adapter, right click and select `Properties`.
Click on `Configure` and open the `Advanced` tab. Search through each of the properties until you find `Large Send Offload Version 2 (IPv4)` and `Large Send Offload Version 2 (IPv6)`. *Disable* both of these properties.

## Advanced Configuration

### How does Ollama handle concurrent requests?

Ollama supports two levels of concurrent processing. If your system has sufficient available memory (system memory when using CPU inference, or VRAM for GPU inference) then multiple models can be loaded at the same time. For a given model, if there is sufficient available memory when the model is loaded, it is configured to allow parallel request processing.

The following server settings may be used to adjust how Ollama handles concurrent requests on most platforms:

- `OLLAMA_MAX_LOADED_MODELS` - The maximum number of models that can be loaded concurrently provided they fit in available memory. The default is 3 * the number of GPUs or 3 for CPU inference.
- `OLLAMA_NUM_PARALLEL` - The maximum number of parallel requests each model will process at the same time. The default will auto-select either 4 or 1 based on available memory.
- `OLLAMA_MAX_QUEUE` - The maximum number of requests Ollama will queue when busy before rejecting additional requests. The default is 512

### How does Ollama load models on multiple GPUs?

When loading a new model, Ollama evaluates the required VRAM for the model against what is currently available. If the model will entirely fit on any single GPU, Ollama will load the model on that GPU. This typically provides the best performance as it reduces the amount of data transferring across the PCI bus during inference. If the model does not fit entirely on one GPU, then it will be spread across all the available GPUs.