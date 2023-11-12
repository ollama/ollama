# Running Ollama on NVIDIA Jetson Devices

With some minor configuration, Ollama runs well on [NVIDIA Jetson Devices](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/). The following has been tested on [JetPack 5.1.2](https://developer.nvidia.com/embedded/jetpack).

NVIDIA Jetson devices are Linux-based embedded AI computers that are purpose-built for AI applications.

Jetsons have an integrated GPU that is wired directly to the memory controller of the machine. For this reason, the `nvidia-smi` command is unrecognized, and Ollama proceeds to operate in "CPU only"
mode. This can be verified by using a monitoring tool like jtop.

In order to address this, we simply pass the path to the Jetson's pre-installed CUDA libraries into `ollama serve` (while in a tmux session). We then hardcode the num_gpu parameters into a cloned
version of our target model.

Prerequisites:

- curl
- tmux

Here are the steps:

- Install Ollama via standard Linux command (ignore the 404 error): `curl https://ollama.ai/install.sh | sh`
- Stop the Ollama service: `sudo systemctl stop ollama`
- Start Ollama serve in a tmux session called ollama_jetson and reference the CUDA libraries path: `tmux has-session -t ollama_jetson 2>/dev/null || tmux new-session -d -s ollama_jetson 
'LD_LIBRARY_PATH=/usr/local/cuda/lib64 ollama serve'`
- Pull the model you want to use (e.g. mistral): `ollama pull mistral`
- Create a new Modelfile specifically for enabling GPU support on the Jetson: `touch ModelfileMistralJetson`
- In the ModelfileMistralJetson file, specify the FROM model and the num_gpu PARAMETER as shown below:

```
FROM mistral
PARAMETER num_gpu 999
```

- Create a new model from your Modelfile: `ollama create mistral-jetson -f ./ModelfileMistralJetson`
- Run the new model: `ollama run mistral-jetson`

If you run a monitoring tool like jtop you should now see that Ollama is using the Jetson's integrated GPU.

And that's it!
