# Linux

## Install

To install Ollama, run the following command:

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

## Manual install

> [!NOTE]
> If you are upgrading from a prior version, you should remove the old libraries with `sudo rm -rf /usr/lib/ollama` first.

Download and extract the package:

```shell
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C /usr -xzf ollama-linux-amd64.tgz
```

Start Ollama:

```shell
ollama serve
```

In another terminal, verify that Ollama is running:

```shell
ollama -v
```

### AMD GPU install

If you have an AMD GPU, also download and extract the additional ROCm package:

```shell
curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tgz -o ollama-linux-amd64-rocm.tgz
sudo tar -C /usr -xzf ollama-linux-amd64-rocm.tgz
```

### ARM64 install

Download and extract the ARM64-specific package:

```shell
curl -L https://ollama.com/download/ollama-linux-arm64.tgz -o ollama-linux-arm64.tgz
sudo tar -C /usr -xzf ollama-linux-arm64.tgz
```

### Adding Ollama as a startup service (recommended)

Create a user and group for Ollama:

```shell
sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
sudo usermod -a -G ollama $(whoami)
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
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
```

Then start the service:

```shell
sudo systemctl daemon-reload
sudo systemctl enable ollama
```

### Install CUDA drivers (optional)

[Download and install](https://developer.nvidia.com/cuda-downloads) CUDA.

Verify that the drivers are installed by running the following command, which should print details about your GPU:

```shell
nvidia-smi
```

### Install AMD ROCm drivers (optional)

[Download and Install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html) ROCm v6.

### Start Ollama

Start Ollama and verify it is running:

```shell
sudo systemctl start ollama
sudo systemctl status ollama
```

> [!NOTE]
> While AMD has contributed the `amdgpu` driver upstream to the official linux
> kernel source, the version is older and may not support all ROCm features. We
> recommend you install the latest driver from
> https://www.amd.com/en/support/linux-drivers for best support of your Radeon
> GPU.

## Customizing

To customize the installation of Ollama, you can edit the systemd service file or the environment variables by running:

```
sudo systemctl edit ollama
```

Alternatively, create an override file manually in `/etc/systemd/system/ollama.service.d/override.conf`:

```ini
[Service]
Environment="OLLAMA_DEBUG=1"
```

## Updating

Update Ollama by running the install script again:

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

Or by re-downloading Ollama:

```shell
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C /usr -xzf ollama-linux-amd64.tgz
```

## Installing specific versions

Use `OLLAMA_VERSION` environment variable with the install script to install a specific version of Ollama, including pre-releases. You can find the version numbers in the [releases page](https://github.com/ollama/ollama/releases).

For example:

```shell
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.3.9 sh
```

## Viewing logs

To view logs of Ollama running as a startup service, run:

```shell
journalctl -e -u ollama
```

## Uninstall

Remove the ollama service:

```shell
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm /etc/systemd/system/ollama.service
```

Remove the ollama binary from your bin directory (either `/usr/local/bin`, `/usr/bin`, or `/bin`):

```shell
sudo rm $(which ollama)
```

Remove the downloaded models and Ollama service user and group:

```shell
sudo rm -r /usr/share/ollama
sudo userdel ollama
sudo groupdel ollama
```
