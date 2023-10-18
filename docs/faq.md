# FAQ

## How can I view the logs?

On macOS:

```
cat ~/.ollama/logs/server.log
```

On Linux:

```
journalctl -u ollama
```

If you're running `ollama serve` directly, the logs will be printed to the console.

## How can I expose the Ollama server?

```bash
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

By default, Ollama allows cross origin requests from `127.0.0.1` and `0.0.0.0`. To support more origins, you can use the `OLLAMA_ORIGINS` environment variable:

```bash
OLLAMA_ORIGINS=http://192.168.1.1:*,https://example.com ollama serve
```

## Where are models stored?

- macOS: Raw model data is stored under `~/.ollama/models`.
- Linux: Raw model data is stored under `/usr/share/ollama/.ollama/models`
