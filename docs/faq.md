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

## How can I expose Ollama on my network?

Ollama binds to 127.0.0.1 port 11434 by default. Change the bind address with the `OLLAMA_HOST` environment variable.

On macOS:

```bash
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

On Linux:

Create a `systemd` drop-in directory and set `Environment=OLLAMA_HOST`

```bash
mkdir -p /etc/systemd/system/ollama.service.d
echo "[Service]" >>/etc/systemd/system/ollama.service.d/environment.conf
```

```bash
echo "Environment=OLLAMA_HOST=0.0.0.0:11434" >>/etc/systemd/system/ollama.service.d/environment.conf
```

Reload `systemd` and restart Ollama:

```bash
systemctl daemon-reload
systemctl restart ollama
```

## How can I allow additional web origins to access Ollama?

Ollama allows cross origin requests from `127.0.0.1` and `0.0.0.0` by default. Add additional origins with the `OLLAMA_ORIGINS` environment variable:

On macOS:

```bash
OLLAMA_ORIGINS=http://192.168.1.1:*,https://example.com ollama serve
```

On Linux:

```bash
echo "Environment=OLLAMA_ORIGINS=http://129.168.1.1:*,https://example.com" >>/etc/systemd/system/ollama.service.d/environment.conf
```

Reload `systemd` and restart Ollama:

```bash
systemctl daemon-reload
systemctl restart ollama
```

## Where are models stored?

- macOS: Raw model data is stored under `~/.ollama/models`.
- Linux: Raw model data is stored under `/usr/share/ollama/.ollama/models`

Below the models directory you will find a structure similar to the following:

```shell
.
├── blobs
└── manifests
   └── registry.ollama.ai
      ├── f0rodo
      ├── library
      ├── mattw
      └── saikatkumardey
```

There is a `manifests/registry.ollama.ai/namespace` path. In example above, the user has downloaded models from the official `library`, `f0rodo`, `mattw`, and `saikatkumardey` namespaces. Within each of those directories, you will find directories for each of the models downloaded. And in there you will find a file name representing each tag. Each tag file is the manifest for the model.  

The manifest lists all the layers used in this model. You will see a `media type` for each layer, along with a digest. That digest corresponds with a file in the `models/blobs directory`.

### How can I change where Ollama stores models?

To modify where models are stored, you can use the `OLLAMA_MODELS` environment variable. Note that on Linux this means defining `OLLAMA_MODELS` in a drop-in `/etc/systemd/system/ollama.service.d` service file, reloading systemd, and restarting the ollama service.

## Does Ollama send my prompts and answers back to Ollama.ai to use in any way?

No. Anything you do with Ollama, such as generate a response from the model, stays with you. We don't collect any data about how you use the model. You are always in control of your own data.

## How can I use Ollama in Visual Studio Code?

There is already a large collection of plugins available for VSCode as well as other editors that leverage Ollama. You can see the list of [extensions & plugins](https://github.com/jmorganca/ollama#extensions--plugins) at the bottom of the main repository readme.

## How do I use Ollama behind a proxy?

Ollama is compatible with proxy servers if `HTTP_PROXY` or `HTTPS_PROXY` are configured. When using either variables, ensure it is set where `ollama serve` can access the values.

When using `HTTPS_PROXY`, ensure the proxy certificate is installed as a system certificate.

On macOS:

```bash
HTTPS_PROXY=http://proxy.example.com ollama serve
```

On Linux:

```bash
echo "Environment=HTTPS_PROXY=https://proxy.example.com" >>/etc/systemd/system/ollama.service.d/environment.conf
```

Reload `systemd` and restart Ollama:

```bash
systemctl daemon-reload
systemctl restart ollama
```

### How do I use Ollama behind a proxy in Docker?

The Ollama Docker container image can be configured to use a proxy by passing `-e HTTPS_PROXY=https://proxy.example.com` when starting the container.

Alternatively, Docker daemon can be configured to use a proxy. Instructions are available for Docker Desktop on [macOS](https://docs.docker.com/desktop/settings/mac/#proxies), [Windows](https://docs.docker.com/desktop/settings/windows/#proxies), and [Linux](https://docs.docker.com/desktop/settings/linux/#proxies), and Docker [daemon with systemd](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy).

Ensure the certificate is installed as a system certificate when using HTTPS. This may require a new Docker image when using a self-signed certificate.

```dockerfile
FROM ollama/ollama
COPY my-ca.pem /usr/local/share/ca-certificates/my-ca.crt
RUN update-ca-certificate
```

Build and run this image:

```shell
docker build -t ollama-with-ca .
docker run -d -e HTTPS_PROXY=https://my.proxy.example.com -p 11434:11434 ollama-with-ca
```
