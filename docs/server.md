# Serving Ollama

By default, using the installation script on Linux, or using the built executable for Mac, `ollama serve` runs automatically. But if you built Ollama yourself or in a few other circumstances, there are reasons you may choose to modify how it runs. On Linux, you may opt to update the Ollama service files, and on all the platforms you may want to run it in a terminal interactively.

## How to use environment variables

Ollama uses environment variables to enable different functionalities. But you need to use them in the right way to get any value from them.

### Using Ollama server environment variables on Mac

On macOS, Ollama runs in the background and is managed by the menubar app. If you want to add environment variables, you will need to manually run Ollama.

1. Click the menubar icon for Ollama and choose **Quit Ollama**.
2. Open a new terminal window and run the following command (this example uses `OLLAMA_HOST` with an IP address of `123.1.1.1`):

   ```bash
   OLLAMA_HOST=123.1.1.1 ollama serve
   ```

### Using Ollama server environment variables on Linux

If you installed Ollama with the install script, a systemd service was created, running as the Ollama user. To add an environment variable, such as OLLAMA_HOST, you will need to follow these steps:

1. Create a `systemd` drop-in directory and add a config file. You only need to do this once.

   ```bash
   mkdir -p /etc/systemd/system/ollama.service.d
   echo '[Service]' >>/etc/systemd/system/ollama.service.d/environment.conf
   ```

2. For each environment variable you wish to use add it to the config file:

   ```bash
   echo 'Environment="OLLAMA_HOST=0.0.0.0:11434"' >>/etc/systemd/system/ollama.service.d/environment.conf
   ```

3. Reload `systemd` and restart Ollama:

   ```bash
   systemctl daemon-reload
   systemctl restart ollama
   ```

## Setting the models directory

When Ollama downloads models, it uses its home directory to store them. This is in a different location on macOS vs. Linux.

- macOS: All model data is stored under `~/.ollama/models`.
- Linux: All model data is stored under `/usr/share/ollama/.ollama/models`

If you want to use a different directory, set the environment variable `OLLAMA_MODELS` to the directory you wish to use. Refer to the section above for how to use environment variables on your platform.

## Exposing Ollama on a Network

Ollama binds to 127.0.0.1 port 11434 by default. Change the bind address with the `OLLAMA_HOST` environment variable. Refer to the section above for how to use environment variables on your platform.

## Allowing additional web origins to access Ollama

Ollama allows cross origin requests from `127.0.0.1` and `0.0.0.0` by default. Add additional origins with the `OLLAMA_ORIGINS` environment variable. For example to add all ports on 192.168.1.1 and https://example.com, you would use:

```shell
OLLAMA_ORIGINS=http://192.168.1.1:*,https://example.com
```

Refer to the section above for how to use environment variables on your platform.

## Using Ollama behind a proxy

Ollama is compatible with proxy servers if `HTTP_PROXY` or `HTTPS_PROXY` are configured. When using either variables, ensure it is set where `ollama serve` can access the values. When using `HTTPS_PROXY`, ensure the proxy certificate is installed as a system certificate. Refer to the section above for how to use environment variables on your platform.

### Using Ollama behind a proxy in Docker

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
