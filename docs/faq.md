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

Ollama binds to 127.0.0.1 port 11434 by default. Change the bind address with the `OLLAMA_HOST` environment variable.

Ollama allows cross origin requests from `127.0.0.1` and `0.0.0.0` by default. Add additional origins with the `OLLAMA_ORIGINS` environment variable:

On macOS:

```bash
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

```bash
OLLAMA_ORIGINS=http://192.168.1.1:*,https://example.com ollama serve
```

On Linux:

Create a `systemd` drop-in directory and set `Environment=OLLAMA_HOST` and/or `Environment=OLLAMA_ORIGINS`

```bash
mkdir -p /etc/systemd/system/ollama.service.d
echo "[Service]" >>/etc/systemd/system/ollama.service.d/environment.conf
```

```bash
echo "Environment=OLLAMA_HOST=0.0.0.0:11434" >>/etc/systemd/system/ollama.service.d/environment.conf
```

```bash
echo "Environment=OLLAMA_HOST=http://129.168.1.1:*,https://example.com" >>/etc/systemd/system/ollama.service.d/environment.conf
```

Reload `systemd` and restart Ollama.

```bash
systemctl daemon-reload
systemctl restart ollama
```

## Where are models stored?

- macOS: Raw model data is stored under `~/.ollama/models`.
- Linux: Raw model data is stored under `/usr/share/ollama/.ollama/models`
