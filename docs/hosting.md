# Serving Ollama

By default, using the installation script on Linux, or using the built executable for Mac, `ollama serve` runs automatically. But if you built Ollama yourself or in a few other circumstances, there are reasons you may choose to modify how it runs. On Linux, you may opt to update the Ollama service files, and on all the platforms you may want to run it in a terminal interactively.

## Exposing Ollama on a Network

Ollama binds to 127.0.0.1 port 11434 by default. Change the bind address with the `OLLAMA_HOST` environment variable.

On macOS:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

On Linux:

Create a `systemd` drop-in directory and set `Environment=OLLAMA_HOST`

```bash
mkdir -p /etc/systemd/system/ollama.service.d
echo '[Service]' >>/etc/systemd/system/ollama.service.d/environment.conf
```

```bash
echo 'Environment="OLLAMA_HOST=0.0.0.0:11434"' >>/etc/systemd/system/ollama.service.d/environment.conf
```

Reload `systemd` and restart Ollama:

```bash
systemctl daemon-reload
systemctl restart ollama
```

## Allowing additional web origins to access Ollama

Ollama allows cross origin requests from `127.0.0.1` and `0.0.0.0` by default. Add additional origins with the `OLLAMA_ORIGINS` environment variable:

On macOS:

```bash
OLLAMA_ORIGINS=http://192.168.1.1:*,https://example.com ollama serve
```

On Linux:

```bash
echo 'Environment="OLLAMA_ORIGINS=http://192.168.1.1:*,https://example.com"' >>/etc/systemd/system/ollama.service.d/environment.conf
```

Reload `systemd` and restart Ollama:

```bash
systemctl daemon-reload
systemctl restart ollama
```

## Using Ollama behind a proxy

Ollama is compatible with proxy servers if `HTTP_PROXY` or `HTTPS_PROXY` are configured. When using either variables, ensure it is set where `ollama serve` can access the values.

When using `HTTPS_PROXY`, ensure the proxy certificate is installed as a system certificate.

On macOS:

```bash
HTTPS_PROXY=http://proxy.example.com ollama serve
```

On Linux:

```bash
echo 'Environment="HTTPS_PROXY=https://proxy.example.com"' >>/etc/systemd/system/ollama.service.d/environment.conf
```

Reload `systemd` and restart Ollama:

```bash
systemctl daemon-reload
systemctl restart ollama
```

### Using Ollama behind a proxy in Docker

The Ollama Docker container image can be configured to use a proxy by passing `-e HTTPS_PROXY=https://proxy.example.com` when starting the container.

Alternatively, Docker daemon can be configured to use a proxy. Instructions are available for Docker Desktop on [macOS](https://docs.docker.com/desktop/settings/mac/#proxies), [Windows](https://docs.docker.com/desktop/settings/windows/#proxies), and [Linux](https://docs.docker.com/desktop/settings/linux/#proxies), and Docker [daemon with systemd](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy).

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