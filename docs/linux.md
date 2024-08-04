# Ollama on Linux

## Install

Install Ollama running this one-liner:

>

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## AMD Radeon GPU support

While AMD has contributed the `amdgpu` driver upstream to the official linux
kernel source, the version is older and may not support all ROCm features. We
recommend you install the latest driver from
https://www.amd.com/en/support/linux-drivers for best support of your Radeon
GPU.

## Manual install

### Download the `ollama` binary

Ollama is distributed as a self-contained binary. Download it to a directory in your PATH according to your CPU architecture:

For Intel x86 architecture:
```bash
sudo curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama
```
For ARM architecure
```bash
sudo curl -L https://ollama.com/download/ollama-linux-arm64 -o /usr/bin/ollama
```

```bash
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

### Install ROCm (optional - for Radeon GPUs)
[Download and Install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)

Make sure to install ROCm v6

### Start Ollama

Start Ollama using `systemd`:

```bash
sudo systemctl start ollama
```

## Update

Update ollama by running the install script again:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Or by downloading the ollama binary:

```bash
sudo curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama
sudo chmod +x /usr/bin/ollama
```

## Installing specific versions

Use `OLLAMA_VERSION` environment variable with the install script to install a specific version of Ollama, including pre-releases. You can find the version numbers in the [releases page](https://github.com/ollama/ollama/releases). 

For example:

```
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.1.32 sh
```

## Viewing logs

To view logs of Ollama running as a startup service, run:

```bash
journalctl -e -u ollama
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

Remove the downloaded models and Ollama service user and group:

```bash
sudo rm -r /usr/share/ollama
sudo userdel ollama
sudo groupdel ollama
```
