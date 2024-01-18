# Development

Install required tools:

- cmake version 3.24 or higher
- go version 1.21 or higher
- gcc version 11.4.0 or higher

```bash
brew install go cmake gcc
```

Optionally enable debugging and more verbose logging:

```bash
# At build time
export CGO_CFLAGS="-g"

# At runtime
export OLLAMA_DEBUG=1
```

Get the required libraries and build the native LLM code:

```bash
go generate ./...
```

Then build ollama:

```bash
go build .
```

Now you can run `ollama`:

```bash
./ollama
```

### Linux

#### Linux CUDA (NVIDIA)

*Your operating system distribution may already have packages for NVIDIA CUDA. Distro packages are often preferable, but instructions are distro-specific. Please consult distro-specific docs for dependencies if available!*

Install `cmake` and `golang` as well as [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) development and runtime packages.
Then generate dependencies:

```
go generate ./...
```

Then build the binary:

```
go build .
```

#### Linux ROCm (AMD)

*Your operating system distribution may already have packages for AMD ROCm and CLBlast. Distro packages are often preferable, but instructions are distro-specific. Please consult distro-specific docs for dependencies if available!*

Install [CLBlast](https://github.com/CNugteren/CLBlast/blob/master/doc/installation.md) and [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) developement packages first, as well as `cmake` and `golang`.
Adjust the paths below (correct for Arch) as appropriate for your distributions install locations and generate dependencies:

```
CLBlast_DIR=/usr/lib/cmake/CLBlast ROCM_PATH=/opt/rocm go generate ./...
```

Then build the binary:

```
go build .
```

ROCm requires elevated privileges to access the GPU at runtime.  On most distros you can add your user account to the `render` group, or run as root.

#### Advanced CPU Settings

By default, running `go generate ./...` will compile a few different variations
of the LLM library based on common CPU families and vector math capabilities,
including a lowest-common-denominator which should run on almost any 64 bit CPU
somewhat slowly.  At runtime, Ollama will auto-detect the optimal variation to
load.  If you would like to build a CPU-based build customized for your
processor, you can set `OLLAMA_CUSTOM_CPU_DEFS` to the llama.cpp flags you would
like to use.  For example, to compile an optimized binary for an Intel i9-9880H,
you might use:

```
OLLAMA_CUSTOM_CPU_DEFS="-DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_F16C=on -DLLAMA_FMA=on" go generate ./...
go build .
```

#### Containerized Linux Build

If you have Docker available, you can build linux binaries with `./scripts/build_linux.sh` which has the CUDA and ROCm dependencies included.  The resulting binary is placed in `./dist`


### Windows

Note: The windows build for Ollama is still under development.

Install required tools:

- MSVC toolchain - C/C++ and cmake as minimal requirements
- go version 1.21 or higher
- MinGW (pick one variant) with GCC.
  - <https://www.mingw-w64.org/>
  - <https://www.msys2.org/>

```powershell
$env:CGO_ENABLED="1"

go generate ./...

go build .
```

#### Windows CUDA (NVIDIA)

In addition to the common Windows development tools described above, install:

- [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
