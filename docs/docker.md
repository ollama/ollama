# Ollama Docker image

### CPU only

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### Nvidia GPU
Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).

#### Install with Apt
1.  Configure the repository
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
```
2.  Install the NVIDIA Container Toolkit packages
```bash
sudo apt-get install -y nvidia-container-toolkit
```

#### Install with Yum or Dnf
1.  Configure the repository

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
```

2. Install the NVIDIA Container Toolkit packages

```bash
sudo yum install -y nvidia-container-toolkit
```

#### Configure Docker to use Nvidia driver
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Start the container

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### JETSON DEVICE

> [!NOTE]  
> If you're running on an NVIDIA JetPack system, Ollama can't automatically discover the correct JetPack version. Pass the environment variable JETSON_JETPACK=5 or JETSON_JETPACK=6 to the container to select version 5 or 6.

Find  JetPack version 
```bash
apt-cache show nvidia-jetpack
```
output like this

```
Package: nvidia-jetpack
Version: 5.1.4-b17
Architecture: arm64
Maintainer: NVIDIA Corporation
Installed-Size: 194
Depends: nvidia-jetpack-runtime (= 5.1.4-b17), nvidia-jetpack-dev (= 5.1.4-b17)
Homepage: http://developer.nvidia.com/jetson
Priority: standard
Section: metapackages
Filename: pool/main/n/nvidia-jetpack/nvidia-jetpack_5.1.4-b17_arm64.deb
Size: 29298
SHA256: 5439dabb8d7a097c215602f7cd11773e4745c5e7d5841d9a0a2551a58b82883e
SHA1: b0c2faa6a9d14e056e394625a11a1a8ed8780327
MD5sum: 05fc4b73de35fd33a535fb887c1947ee
Description: NVIDIA Jetpack Meta Package
Description-md5: ad1462289bdbc54909ae109d1d32c0a8
```
set env variable for JetPack version when running docker 
also use --runtime=nvidia on jetson not  --gpus 

#### Start the container
```bash
docker run -d --runtime=nvidia -e JETSON_JETPACK=JetPack-version -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```


### AMD GPU

To run Ollama using Docker with AMD GPUs, use the `rocm` tag and the following command:

```
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
```

### Run model locally

Now you can run a model:

```
docker exec -it ollama ollama run llama3.2
```

### Try different models

More models can be found on the [Ollama library](https://ollama.com/library).
