# Configuration

There are several places Ollama can be configured:

- Environment variables (see [table](#environment-variables) below)
- In API requests (see [api](api.md) and [openai](openai.md))
- In the CLI client using the `/set` command (see [faq](faq.md))
- Modelfile parameters using `PARAMETER` statements (see [Modelfile](modelfile.md))
- At build time (e.g. `OLLAMA_CUSTOM_CPU_DEFS`, see [development](development.md))

## Environment Variables

The following environment variables can be used to configure Ollama:

| Variable                   | Client/Server | Description                                                                                     | Example                     |
| -------------------------- | ------------- | ----------------------------------------------------------------------------------------------- | --------------------------- |
| `OLLAMA_HOST`              | Client/Server | IP Address for the ollama server                                                                | `0.0.0.0`                   |
| `OLLAMA_NOHISTORY`         | Client/Server | Do not preserve [readline](https://en.wikipedia.org/wiki/GNU_Readline) history                  | `true`                      |
| `OLLAMA_DEBUG`             | Server        | Show additional debug information                                                               | `true`                      |
| `OLLAMA_FLASH_ATTENTION`   | Server        | Enable [flash attention](https://github.com/ggerganov/llama.cpp/pull/5021)                      | `true`                      |
| `OLLAMA_KEEP_ALIVE`        | Server        | The duration that models stay loaded in memory                                                  | `10m`                       |
| `OLLAMA_LLM_LIBRARY`       | Server        | Set LLM library to bypass autodetection                                                         | `cuda_v12`                  |
| `OLLAMA_MAX_LOADED_MODELS` | Server        | Maximum number of loaded models per GPU (each model can perform `OLLAMA_NUM_PARALLEL` requests) | `2`                         |
| `OLLAMA_MAX_QUEUE`         | Server        | Maximum number of queued requests                                                               | `100`                       |
| `OLLAMA_MAX_VRAM`          | Server        | Maximum VRAM Ollama may use                                                                     | `8192`                      |
| `OLLAMA_MODELS`            | Server        | The path to the models directory                                                                | `/home/user/.ollama/models` |
| `OLLAMA_NOPRUNE`           | Server        | Do not prune model blobs on startup                                                             | `true`                      |
| `OLLAMA_NUM_PARALLEL`      | Server        | Maximum number of parallel requests (effectively consumes n* the context/VRAM)                  | `2`                         |
| `OLLAMA_ORIGINS`           | Server        | A comma separated list of allowed origins                                                       | `127.0.0.1,0.0.0.0`         |
| `OLLAMA_RUNNERS_DIR`       | Server        | Location for runners                                                                            | `/tmp/ollama_runners`       |
| `OLLAMA_SCHED_SPREAD`      | Server        | Always schedule model across all GPUs                                                           | `true`                      |
| `OLLAMA_TMPDIR`            | Server        | Location for temporary files                                                                    | `/tmp/ollama_tmp`           |
| `OLLAMA_INTEL_GPU`         | Server        | Enable experimental Intel GPU detection                                                         | `true`                      |
| ---                        | ---           | ---                                                                                             | ---                         |
| `CUDA_VISIBLE_DEVICES`     | Server        | Set which NVIDIA devices are visible                                                            | `0,1`                       |
| `HIP_VISIBLE_DEVICES`      | Server        | Set which AMD devices are visible                                                               | `0,1`                       |
| `ROCR_VISIBLE_DEVICES`     | Server        | Set which AMD devices are visible                                                               | `0,1`                       |
| `GPU_DEVICE_ORDINAL`       | Server        | Set which AMD devices are visible                                                               | `0,1`                       |
| `HSA_OVERRIDE_GFX_VERSION` | Server        | Override the gfx Server used for all detected AMD GPUs                                          | `gfx1100`                   |

_Note that the examples given above are not the default values._

Many runtime environment variables can also be listed on the command line client by passing `--help` to commands (e.g. `ollama serve --help`).

### Setting Environment Variables

How you're running Ollama will determine how you set these variables.

For example, if you're running Ollama in:

- Docker (see [docker](docker.md))
  - When using `docker run`, you can set these variables with the `-e` flag.
  - When using docker compose under the `environment` key, or in a configured ("`env_files:`") `.env` file.
- Linux / macOS (see [faq](faq.md))
  - Using `export` in your shell if you're running Ollama from the command line, e.g. `OLLAMA_HOST="0.0.0.0" ollama serve llama3`.
- Linux (see [linux](linux.md))
  - Using `systemd` service files if you're running Ollama as a service with systemd, e.g. `systemctl edit ollama.service`.
- MacOS
  - Setting the launchctl environment variables if you're running Ollama as a service `e.g. launchctl setenv OLLAMA_HOST "0.0.0.0"`
- Windows (see [windows](windows.md))
  - Using `set` in your shell if you're running Ollama from the command line, e.g. `set OLLAMA_HOST=0.0.0.0 ollama serve llama3`.

### Development

See [development](development.md).

An up to date listing of available environment variables can be found in[envconfig/config.go](https://github.com/ollama/ollama/blob/main/envconfig/config.go)
