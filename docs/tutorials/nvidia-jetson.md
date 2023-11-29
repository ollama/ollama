# Running Ollama on NVIDIA Jetson Devices

With some minor configuration, Ollama runs well on [NVIDIA Jetson Devices](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/). The following has been tested on [JetPack 5.1.2](https://developer.nvidia.com/embedded/jetpack).

NVIDIA Jetson devices are Linux-based embedded AI computers that are purpose-built for AI applications.

Jetsons have an integrated GPU that is wired directly to the memory controller of the machine. For this reason, the `nvidia-smi` command is unrecognized, and Ollama proceeds to operate in "CPU only" mode. This can be verified by using a monitoring tool like jtop.

In order to address this, we simply pass the path to the Jetson's pre-installed CUDA libraries into `ollama serve`. We then hardcode the num_gpu parameters into a cloned version of our target model.

## Step-by-step process

### Update the system package lists

```
sudo apt update
```

### Install `curl` (if it's not already installed)

```
sudo apt install curl
```

### Run the installation script for Ollama

```
curl https://ollama.ai/install.sh | sh
```

### Stop the Ollama service (if it's already running)

```
sudo systemctl stop ollama
```

### Start the Ollama service in the background and redirect its output to a log file

```
nohup bash -c 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 ollama serve' > ollama_serve.log 2>&1 &
```

### Pull the Mistral model

```
ollama pull mistral
```

### Create a `Modelfile` for Mistral on Jetson (specifying a num_gpu PARAMETER)

```
echo -e 'FROM mistral\nPARAMETER num_gpu 999' > ModelfileMistralJetson
```

### Create a new `mistral-jetson` model using the new `Modelfile`

```
ollama create mistral-jetson -f ./ModelfileMistralJetson
```

### Run the `mistral-jetson` model

```
ollama run mistral-jetson
```

If you run a monitoring tool like jtop you should now see that Ollama is using the Jetson's integrated GPU.

And that's it!

## Quickstart snippet

The above commands have been packaged into a single snippet that is designed to get anyone up and running on a Jetson device with at least 8GB of RAM (e.g. a Jetson Orin Developer Kit).

The below should be suitable for getting anyone up and running on a freshly flashed Micro SD card with JetPack 5.1.2 following [these instructions](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit#write).

Here's the snippet to execute once you're up and running with the device.

```
sudo apt update && \
sudo apt install curl && \
curl -f https://ollama.ai/install.sh | sh || true && \
sudo systemctl stop ollama && \
nohup bash -c 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 ollama serve' > ollama_serve.log 2>&1 & \
ollama pull mistral && \
echo -e 'FROM mistral\nPARAMETER num_gpu 999' > ModelfileMistralJetson && \
ollama create mistral-jetson -f ./ModelfileMistralJetson && \
ollama run mistral-jetson
```
