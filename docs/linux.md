# Ollama on Linux

## Install

Install Ollama running this one-liner:
>
```bash
curl https://ollama.ai/install.sh | sh
```

## Manual install

### Download the `ollama` binary

Ollama is distributed as a self-contained binary. Download it to a directory in your PATH:

```bash
sudo curl -L https://ollama.ai/download/ollama-linux-amd64 -o /usr/bin/ollama
sudo chmod +x /usr/bin/ollama
```

### Adding Ollama as a startup service (recommended)

Create a user for Ollama:

```bash
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

[Install]
WantedBy=default.target
```

Then start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama
```

### Install CUDA drivers (optional – for Nvidia GPUs)

[Download and install](https://developer.nvidia.com/cuda-downloads) CUDA.

Verify that the drivers are installed by running the following command, which should print details about your GPU:

```bash
nvidia-smi
```

### Start Ollama

Start Ollama using `systemd`:

```bash
sudo systemctl start ollama
```

## Update

Update ollama by running the install script again:

```bash
curl https://ollama.ai/install.sh | sh
```

Or by downloading the ollama binary:

```bash
sudo curl -L https://ollama.ai/download/ollama-linux-amd64 -o /usr/bin/ollama
sudo chmod +x /usr/bin/ollama
```

## Viewing logs

To view logs of Ollama running as a startup service, run:

```bash
journalctl -u ollama
```

## Uninstall

Remove the ollama service:

```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm /etc/systemd/system/ollama.service
```

Remove the ollama binary from your bin directory (either `/usr/local/bin`, `/usr/bin`, or `/bin`):

```bash
sudo rm $(which ollama)
```

Remove the downloaded models and Ollama service user:
```bash
sudo rm -r /usr/share/ollama
sudo userdel ollama
```
