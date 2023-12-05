# Running Ollama on NVIDIA Jetson Devices

With some minor configuration, Ollama runs well on [NVIDIA Jetson Devices](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/). The following has been tested on [JetPack 5.1.2](https://developer.nvidia.com/embedded/jetpack).

NVIDIA Jetson devices are Linux-based embedded AI computers that are purpose-built for AI applications.

Jetsons have an integrated GPU that is wired directly to the memory controller of the machine. For this reason, the `nvidia-smi` command is unrecognized, and Ollama proceeds to operate in "CPU only" mode. This can be verified by using a monitoring tool like jtop.

In order to address this, we simply pass the path to the Jetson's pre-installed CUDA libraries into `ollama serve`. We also need to specify a num_gpu parameter (either within the REPL or in a cloned version of our target model).

The below example uses Mistral and uses a num_gpu value of 999 (which can be tailored depending on the target model as needed).

## Update the system package lists

```
sudo apt update
```

## Install `curl` (if it's not already installed)

```
sudo apt install curl
```

## Run the installation script for Ollama (ignore the NVIDIA 404)

```
curl https://ollama.ai/install.sh | sh
```

## Stop the Ollama service (if it's already running)

```
sudo systemctl stop ollama
```

## Start the Ollama service in the background and redirect its output to a log file

```
nohup bash -c 'LD_LIBRARY_PATH=/usr/local/cuda/lib64 ollama serve' > ollama_serve.log 2>&1 &
```

## Pull the Mistral model

```
ollama pull mistral
```

## Set the num_gpu parameter so that the Jetson's integrated GPU is used (two options as outlined below)

### Option 1: Set the num_gpu parameter within the REPL

```
ollama run mistral # Run the model and wait for it to load
/set parameter num_gpu 999 # Set the parameter within the REPL
```

### Option 2: Set the num_gpu parameter in a `Modelfile` (called mistral-jetson)

```
echo -e 'FROM mistral\nPARAMETER num_gpu 999' > ModelfileMistralJetson # Create a Modelfile containing the parameter
ollama create mistral-jetson -f ./ModelfileMistralJetson # Create a model from the Modelfile
ollama run mistral-jetson # Run the new model
```

If you run a monitoring tool like jtop you should now see that Ollama is using the Jetson's integrated GPU.

And that's it!
