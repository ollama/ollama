# How to troubleshoot issues

Sometimes Ollama may not perform as expected. One of the best ways to figure out what happened is to take a look at the logs. Find the logs on Mac by running the command:

```shell
cat ~/.ollama/logs/server.log
```

On Linux systems with systemd, the logs can be found with this command:

```shell
journalctl -u ollama
```

If manually running `ollama serve` in a terminal, the logs will be on that terminal.

If using Docker, logs can be found by running `docker logs <container name>`:
```shell
docker logs ollama
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

## Known issues

* N/A
