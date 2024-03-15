# GPU
## Nvidia
Ollama supports Nvidia GPUs with compute capability 5.0 to 8.6.

Check your compute compatibility to see if your card is supported:
[https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)

| GPU Name                | Compute Capability |
|-------------------------|--------------------|
| GeForce RTX 4090        | 8.9                |
| GeForce RTX 4080        | 8.9                |
| GeForce RTX 4070 Ti     | 8.9                |
| GeForce RTX 4060 Ti     | 8.9                |
| GeForce RTX 3090 Ti     | 8.6                |
| GeForce RTX 3090        | 8.6                |
| GeForce RTX 3080 Ti     | 8.6                |
| GeForce RTX 3080        | 8.6                |
| GeForce RTX 3070 Ti     | 8.6                |
| GeForce RTX 3070        | 8.6                |
| GeForce RTX 3060 Ti     | 8.6                |
| GeForce RTX 3060        | 8.6                |
| GeForce GTX 1650 Ti     | 7.5                |
| NVIDIA TITAN RTX        | 7.5                |
| GeForce RTX 2080 Ti     | 7.5                |
| GeForce RTX 2080        | 7.5                |
| GeForce RTX 2070        | 7.5                |
| GeForce RTX 2060        | 7.5                |
| NVIDIA TITAN V          | 7.0                |
| NVIDIA TITAN Xp         | 6.1                |
| NVIDIA TITAN X          | 6.1                |
| GeForce GTX 1080 Ti     | 6.1                |
| GeForce GTX 1080        | 6.1                |
| GeForce GTX 1070 Ti     | 6.1                |
| GeForce GTX 1070        | 6.1                |
| GeForce GTX 1060        | 6.1                |
| GeForce GTX 1050        | 6.1                |
| GeForce GTX TITAN X     | 5.2                |
| GeForce GTX 980 Ti      | 5.2                |
| GeForce GTX 980         | 5.2                |
| GeForce GTX 970         | 5.2                |
| GeForce GTX 960         | 5.2                |
| GeForce GTX 950         | 5.2                |
| GeForce GTX 750 Ti      | 5.0                |
| GeForce GTX 750         | 5.0                |
| NVIDIA NVS 810          | 5.0                |
| NVIDIA H100             | 9.0                |
| NVIDIA L4               | 8.9                |
| NVIDIA L40              | 8.9                |
| RTX 6000                | 8.9                |
| NVIDIA A100             | 8.0                |
| NVIDIA A40              | 8.6                |
| NVIDIA A30              | 8.0                |
| NVIDIA A10              | 8.6                |
| NVIDIA A16              | 8.6                |
| NVIDIA A2               | 8.6                |
| RTX A6000               | 8.6                |
| RTX A5000               | 8.6                |
| RTX A4000               | 8.6                |
| RTX A3000               | 8.6                |
| RTX A2000               | 8.6                |
| NVIDIA T4               | 7.5                |
| RTX 5000                | 7.5                |
| RTX 4000                | 7.5                |
| RTX 3000                | 7.5                |
| T2000                   | 7.5                |
| T1200                   | 7.5                |
| T1000                   | 7.5                |
| T600                    | 7.5                |
| T500                    | 7.5                |
| Quadro RTX 8000         | 7.5                |
| Quadro RTX 6000         | 7.5                |
| Quadro RTX 5000         | 7.5                |
| Quadro RTX 4000         | 7.5                |
| NVIDIA V100             | 7.0                |
| Quadro GV100            | 7.0                |
| Tesla P100              | 6.0                |
| Quadro GP100            | 6.0                |
| Tesla P40               | 6.1                |
| Tesla P4                | 6.1                |
| Quadro P6000            | 6.1                |
| Quadro P5200            | 6.1                |
| Quadro P4200            | 6.1                |
| Quadro P3200            | 6.1                |
| Quadro P5000            | 6.1                |
| Quadro P4000            | 6.1                |
| Quadro P3000            | 6.1                |
| Quadro P2200            | 6.1                |
| Quadro P2000            | 6.1                |
| Quadro P1000            | 6.1                |
| Quadro P620             | 6.1                |
| Quadro P600             | 6.1                |
| Quadro P500             | 6.1                |
| Quadro P520             | 6.1                |
| Tesla M60               | 5.2                |
| Tesla M40               | 5.2                |
| Quadro M6000 24GB       | 5.2                |
| Quadro M6000            | 5.2                |
| Quadro M5000            | 5.2                |
| Quadro M5500M           | 5.2                |
| Quadro M4000            | 5.2                |
| Quadro M2200            | 5.2                |
| Quadro M2000            | 5.2                |
| Quadro M620             | 5.2                |
| Quadro K2200            | 5.0                |
| Quadro K1200            | 5.0                |
| Quadro K620             | 5.0                |
| Quadro M1200            | 5.0                |
| Quadro M520             | 5.0                |
| Quadro M5000M           | 5.0                |
| Quadro M4000M           | 5.0                |
| Quadro M3000M           | 5.0                |
| Quadro M2000M           | 5.0                |
| Quadro M1000M           | 5.0                |
| Quadro K620M            | 5.0                |
| Quadro M600M            | 5.0                |
| Quadro M500M            | 5.0                |

## AMD Radeon

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

### Metal (Apple GPUs)
Ollama supports GPU acceleration on Apple devices via the Metal API.