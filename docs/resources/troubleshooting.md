# Troubleshooting Guide

This guide provides solutions for common issues you might encounter when using Ollama.

## Finding Logs

Logs are essential for diagnosing issues. Here's how to find them on different platforms:

### macOS

```shell
cat ~/.ollama/logs/server.log
```

### Linux (with systemd)

```shell
journalctl -u ollama --no-pager --follow --pager-end 
```

### Docker Container

```shell
docker logs <container-name>
```

(Use `docker ps` to find the container name)

### Windows

You can view logs in the explorer window by hitting `<cmd>+R` and typing:
- `explorer %LOCALAPPDATA%\Ollama` to view logs. The most recent server logs will be in `server.log` and older logs will be in `server-#.log`
- `explorer %LOCALAPPDATA%\Programs\Ollama` to browse the binaries (The installer adds this to your user PATH)
- `explorer %HOMEPATH%\.ollama` to browse where models and configuration are stored

To enable additional debug logging on Windows, first **Quit the running app from the tray menu** then in a PowerShell terminal:

```powershell
$env:OLLAMA_DEBUG="1"
& "ollama app.exe"
```

## LLM Libraries

Ollama includes multiple LLM libraries compiled for different GPUs and CPU vector features. Ollama tries to pick the best one based on the capabilities of your system. If this autodetection has problems, or you run into other issues (e.g., crashes in your GPU), you can work around this by forcing a specific LLM library.

In the server log, you will see a message that looks something like this (varies from release to release):

```
Dynamic LLM libraries [rocm_v6 cpu cpu_avx cpu_avx2 cuda_v11 rocm_v5]
```

### Experimental LLM Library Override

You can set `OLLAMA_LLM_LIBRARY` to any of the available LLM libraries to bypass autodetection. For example, if you have a CUDA card but want to force the CPU LLM library with AVX2 vector support, use:

```shell
OLLAMA_LLM_LIBRARY="cpu_avx2" ollama serve
```

You can see what features your CPU has with the following command:

```shell
cat /proc/cpuinfo| grep flags | head -1
```

## Installing Older or Pre-release Versions on Linux

If you run into problems on Linux and want to install an older version, or you'd like to try out a pre-release before it's officially released, you can tell the install script which version to install:

```shell
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.5.7 sh
```

## GPU Issues

### Linux Docker

If Ollama initially works on the GPU in a Docker container but then switches to running on CPU after some period of time with errors in the server log reporting GPU discovery failures, this can be resolved by disabling systemd cgroup management in Docker. Edit `/etc/docker/daemon.json` on the host and add `"exec-opts": ["native.cgroupdriver=cgroupfs"]` to the Docker configuration.

### NVIDIA GPU Discovery

When Ollama starts up, it takes inventory of the GPUs present in the system to determine compatibility and how much VRAM is available. Sometimes this discovery can fail to find your GPUs. In general, running the latest driver will yield the best results.

#### Linux NVIDIA Troubleshooting

If you are using a container to run Ollama, make sure you've set up the container runtime first as described in the [Docker documentation](../installation/docker.md).

Sometimes Ollama can have difficulties initializing the GPU. When you check the server logs, this can show up as various error codes, such as "3" (not initialized), "46" (device unavailable), "100" (no device), "999" (unknown), or others. The following troubleshooting techniques may help resolve the problem:

- If you are using a container, is the container runtime working? Try `docker run --gpus all ubuntu nvidia-smi` - if this doesn't work, Ollama won't be able to see your NVIDIA GPU.
- Is the uvm driver loaded? `sudo nvidia-modprobe -u`
- Try reloading the nvidia_uvm driver - `sudo rmmod nvidia_uvm` then `sudo modprobe nvidia_uvm`
- Try rebooting
- Make sure you're running the latest NVIDIA drivers

If none of those resolve the problem, gather additional information and file an issue:
- Set `CUDA_ERROR_LEVEL=50` and try again to get more diagnostic logs
- Check dmesg for any errors: `sudo dmesg | grep -i nvrm` and `sudo dmesg | grep -i nvidia`

### AMD GPU Discovery

On Linux, AMD GPU access typically requires `video` and/or `render` group membership to access the `/dev/kfd` device. If permissions are not set up correctly, Ollama will detect this and report an error in the server log.

When running in a container, in some Linux distributions and container runtimes, the Ollama process may be unable to access the GPU. Use `ls -lnd /dev/kfd /dev/dri /dev/dri/*` on the host system to determine the **numeric** group IDs on your system, and pass additional `--group-add ...` arguments to the container so it can access the required devices. For example, in the following output `crw-rw---- 1 0 44 226, 0 Sep 16 16:55 /dev/dri/card0` the group ID column is `44`.

If you are experiencing problems getting Ollama to correctly discover or use your GPU for inference, the following may help isolate the failure:
- `AMD_LOG_LEVEL=3` Enable info log levels in the AMD HIP/ROCm libraries. This can help show more detailed error codes that can help troubleshoot problems.
- `OLLAMA_DEBUG=1` During GPU discovery, additional information will be reported.
- Check dmesg for any errors from amdgpu or kfd drivers: `sudo dmesg | grep -i amdgpu` and `sudo dmesg | grep -i kfd`

### Multiple AMD GPUs

If you experience gibberish responses when models load across multiple AMD GPUs on Linux, see the [AMD ROCm documentation](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/mgpu.html#mgpu-known-issues-and-limitations).

## Windows Terminal Errors

Older versions of Windows 10 (e.g., 21H1) are known to have a bug where the standard terminal program does not display control characters correctly. This can result in a long string of strings like `←[?25h←[?25l` being displayed, sometimes erroring with `The parameter is incorrect`. To resolve this problem, please update to Windows 10 22H1 or newer.

## Getting Help

If you're still experiencing issues after trying these troubleshooting steps, join the [Ollama Discord](https://discord.gg/ollama) for help interpreting the logs and additional support.