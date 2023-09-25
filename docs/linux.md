# Installing Ollama on Linux

> Note: A one line installer for Ollama is available by running:
>
> ```
> curl https://ollama.ai/install.sh | sh
> ```

## Download the `ollama` binary

Ollama is distributed as a self-contained binary. Download it to a directory in your PATH:

```
sudo curl -L https://ollama.ai/download/ollama-linux-amd64 -o /usr/bin/ollama
sudo chmod +x /usr/bin/ollama
```

## Start Ollama

Start Ollama by running `ollama serve`:

```
ollama serve
```

Once Ollama is running, run a model in another terminal session:

```
ollama run llama2
```

## Install CUDA drivers (optional – for Nvidia GPUs)

[Download and install](https://developer.nvidia.com/cuda-downloads) CUDA.

Verify that the drivers are installed by running the following command, which should print details about your GPU:

```
nvidia-smi
```

## Adding Ollama as a startup service (optional)

Create a user for Ollama:

```
sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama
```

Create a service file in `/etc/systemd/system/ollama.service`:

```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="HOME=/usr/share/ollama"

[Install]
WantedBy=default.target
```

Then start the service:

```
sudo systemctl daemon-reload
sudo systemctl enable ollama
```

### Viewing logs

To view logs of Ollama running as a startup service, run:

```
journalctl -u ollama
```

