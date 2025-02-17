# Ollama Docker image

### CPU only

```shell
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### Nvidia GPU
Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).

#### Install with Apt
1.  Configure the repository

    ```shell
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    ```

2.  Install the NVIDIA Container Toolkit packages

    ```shell
    sudo apt-get install -y nvidia-container-toolkit
    ```

#### Install with Yum or Dnf
1.  Configure the repository

    ```shell
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
        | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
    ```

2. Install the NVIDIA Container Toolkit packages

    ```shell
    sudo yum install -y nvidia-container-toolkit
    ```

#### Configure Docker to use Nvidia driver

```shell
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Start the container

```shell
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

> [!NOTE]  
> If you're running on an NVIDIA JetPack system, Ollama can't automatically discover the correct JetPack version. Pass the environment variable JETSON_JETPACK=5 or JETSON_JETPACK=6 to the container to select version 5 or 6.

### AMD GPU

To run Ollama using Docker with AMD GPUs, use the `rocm` tag and the following command:

```shell
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
```

### Run model locally

Now you can run a model:

```shell
docker exec -it ollama ollama run llama3.2
```

### Try different models

More models can be found on the [Ollama library](https://ollama.com/library).
