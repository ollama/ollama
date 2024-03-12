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

When you run Ollama on **Windows**, there are a few different locations.  You can view them in the explorer window by hitting `<cmd>+R` and type in:
- `explorer %LOCALAPPDATA%\Ollama` to view logs
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

Ollama includes multiple LLM libraries compiled for different GPUs and CPU
vector features.  Ollama tries to pick the best one based on the capabilities of
your system.  If this autodetection has problems, or you run into other problems
(e.g. crashes in your GPU) you can workaround this by forcing a specific LLM
library.  `cpu_avx2` will perform the best, followed by `cpu_avx` an the slowest
but most compatible is `cpu`.  Rosetta emulation under MacOS will work with the
`cpu` library. 

In the server log, you will see a message that looks something like this (varies
from release to release):

```
Dynamic LLM libraries [rocm_v6 cpu cpu_avx cpu_avx2 cuda_v11 rocm_v5]
```

**Experimental LLM Library Override**

You can set OLLAMA_LLM_LIBRARY to any of the available LLM libraries to bypass
autodetection, so for example, if you have a CUDA card, but want to force the
CPU LLM library with AVX2 vector support, use:

```
OLLAMA_LLM_LIBRARY="cpu_avx2" ollama serve
```

You can see what features your CPU has with the following.  
```
cat /proc/cpuinfo| grep flags  | head -1
```

## AMD Radeon GPU Support

Ollama leverages the AMD ROCm library, which does not support all AMD GPUs. In
some cases you can force the system to try to use a similar LLVM target that is
close.  For example The Radeon RX 5400 is `gfx1034` (also known as 10.3.4)
however, ROCm does not currently support this target. The closest support is
`gfx1030`.  You can use the environment variable `HSA_OVERRIDE_GFX_VERSION` with
`x.y.z` syntax.  So for example, to force the system to run on the RX 5400, you
would set `HSA_OVERRIDE_GFX_VERSION="10.3.0"` as an environment variable for the
server.  If you have an unsupported AMD GPU you can experiment using the list of
supported types below.

At this time, the known supported GPU types are the following LLVM Targets.
This table shows some example GPUs that map to these LLVM targets:
| **LLVM Target** | **An Example GPU** |
|-----------------|---------------------|
| gfx900 | Radeon RX Vega 56 |
| gfx906 | Radeon Instinct MI50 |
| gfx908 | Radeon Instinct MI100 |
| gfx90a | Radeon Instinct MI210 |
| gfx940 | Radeon Instinct MI300 |
| gfx941 | |
| gfx942 | |
| gfx1030 | Radeon PRO V620 |
| gfx1100 | Radeon PRO W7900 |
| gfx1101 | Radeon PRO W7700 |
| gfx1102 | Radeon RX 7600 |

AMD is working on enhancing ROCm v6 to broaden support for families of GPUs in a
future release which should increase support for more GPUs.

Reach out on [Discord](https://discord.gg/ollama) or file an
[issue](https://github.com/ollama/ollama/issues) for additional help.

## Installing older versions on Linux

If you run into problems on Linux and want to install an older version you can tell the install script
which version to install.

```sh
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="0.1.27" sh
```
