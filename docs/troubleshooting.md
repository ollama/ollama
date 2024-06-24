# How to troubleshoot issues

Sometimes Ollama may not perform as expected. One of the best ways to figure out what happened is to take a look at the logs. Find the logs on **Mac** by running the command:

```shell
cat ~/.ollama/logs/server.log
```

On **Linux** systems with systemd, the logs can be found with this command:

```shell
journalctl -u ollama
```

When you run Ollama in a **container**, the logs go to stdout/stderr in the container:

```shell
docker logs <container-name>
```
(Use `docker ps` to find the container name)

If manually running `ollama serve` in a terminal, the logs will be on that terminal.

When you run Ollama on **Windows**, there are a few different locations. You can view them in the explorer window by hitting `<cmd>+R` and type in:
- `explorer %LOCALAPPDATA%\Ollama` to view logs.  The most recent server logs will be in `server.log` and older logs will be in `server-#.log` 
- `explorer %LOCALAPPDATA%\Programs\Ollama` to browse the binaries (The installer adds this to your user PATH)
- `explorer %HOMEPATH%\.ollama` to browse where models and configuration is stored
- `explorer %TEMP%` where temporary executable files are stored in one or more `ollama*` directories

To enable additional debug logging to help troubleshoot problems, first **Quit the running app from the tray menu** then in a powershell terminal
```powershell
$env:OLLAMA_DEBUG="1"
& "ollama app.exe"
```

Join the [Discord](https://discord.gg/ollama) for help interpreting the logs.

## LLM libraries

Ollama includes multiple LLM libraries compiled for different GPUs and CPU vector features. Ollama tries to pick the best one based on the capabilities of your system. If this autodetection has problems, or you run into other problems (e.g. crashes in your GPU) you can workaround this by forcing a specific LLM library. `cpu_avx2` will perform the best, followed by `cpu_avx` an the slowest but most compatible is `cpu`. Rosetta emulation under MacOS will work with the `cpu` library. 

In the server log, you will see a message that looks something like this (varies from release to release):

```
Dynamic LLM libraries [rocm_v6 cpu cpu_avx cpu_avx2 cuda_v11 rocm_v5]
```

**Experimental LLM Library Override**

You can set OLLAMA_LLM_LIBRARY to any of the available LLM libraries to bypass autodetection, so for example, if you have a CUDA card, but want to force the CPU LLM library with AVX2 vector support, use:

```
OLLAMA_LLM_LIBRARY="cpu_avx2" ollama serve
```

You can see what features your CPU has with the following.
```
cat /proc/cpuinfo| grep flags | head -1
```

## Installing older or pre-release versions on Linux

If you run into problems on Linux and want to install an older version, or you'd like to try out a pre-release before it's officially released, you can tell the install script which version to install.

```sh
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="0.1.29" sh
```

## Linux tmp noexec 

If your system is configured with the "noexec" flag where Ollama stores its temporary executable files, you can specify an alternate location by setting OLLAMA_TMPDIR to a location writable by the user ollama runs as. For example OLLAMA_TMPDIR=/usr/share/ollama/

## Container fails to run on NVIDIA GPU

Make sure you've set up the container runtime first as described in [docker.md](./docker.md)

Sometimes the container runtime can have difficulties initializing the GPU. When you check the server logs, this can show up as various error codes, such as "3" (not initialized), "46" (device unavailable), "100" (no device), "999" (unknown), or others. The following troubleshooting techniques may help resolve the problem

- Is the container runtime working?  Try `docker run --gpus all ubuntu nvidia-smi` - if this doesn't work, Ollama wont be able to see your NVIDIA GPU.
- Is the uvm driver not loaded? `sudo nvidia-modprobe -u`
- Try reloading the nvidia_uvm driver - `sudo rmmod nvidia_uvm` then `sudo modprobe nvidia_uvm`
- Try rebooting
- Make sure you're running the latest nvidia drivers

If none of those resolve the problem, gather additional information and file an issue:
- Set `CUDA_ERROR_LEVEL=50` and try again to get more diagnostic logs
- Check dmesg for any errors `sudo dmesg | grep -i nvrm` and `sudo dmesg | grep -i nvidia`
