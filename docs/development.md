# Development

Install required tools:

- cmake version 3.24 or higher
- go version 1.22 or higher
- gcc version 11.4.0 or higher

### MacOS

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

_Your operating system distribution may already have packages for NVIDIA CUDA. Distro packages are often preferable, but instructions are distro-specific. Please consult distro-specific docs for dependencies if available!_

Install `cmake` and `golang` as well as [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
development and runtime packages.

Typically the build scripts will auto-detect CUDA, however, if your Linux distro
or installation approach uses unusual paths, you can specify the location by
specifying an environment variable `CUDA_LIB_DIR` to the location of the shared
libraries, and `CUDACXX` to the location of the nvcc compiler. You can customize
a set of target CUDA architectures by setting `CMAKE_CUDA_ARCHITECTURES` (e.g. "50;60;70")

Then generate dependencies:

```
go generate ./...
```

Then build the binary:

```
go build .
```

#### Linux ROCm (AMD)

_Your operating system distribution may already have packages for AMD ROCm and CLBlast. Distro packages are often preferable, but instructions are distro-specific. Please consult distro-specific docs for dependencies if available!_

Install [CLBlast](https://github.com/CNugteren/CLBlast/blob/master/doc/installation.md) and [ROCm](https://rocm.docs.amd.com/en/latest/) development packages first, as well as `cmake` and `golang`.

Typically the build scripts will auto-detect ROCm, however, if your Linux distro
or installation approach uses unusual paths, you can specify the location by
specifying an environment variable `ROCM_PATH` to the location of the ROCm
install (typically `/opt/rocm`), and `CLBlast_DIR` to the location of the
CLBlast install (typically `/usr/lib/cmake/CLBlast`). You can also customize
the AMD GPU targets by setting AMDGPU_TARGETS (e.g. `AMDGPU_TARGETS="gfx1101;gfx1102"`)

```
go generate ./...
```

Then build the binary:

```
go build .
```

ROCm requires elevated privileges to access the GPU at runtime. On most distros you can add your user account to the `render` group, or run as root.

#### Advanced CPU Settings

By default, running `go generate ./...` will compile a few different variations
of the LLM library based on common CPU families and vector math capabilities,
including a lowest-common-denominator which should run on almost any 64 bit CPU
somewhat slowly. At runtime, Ollama will auto-detect the optimal variation to
load. If you would like to build a CPU-based build customized for your
processor, you can set `OLLAMA_CUSTOM_CPU_DEFS` to the llama.cpp flags you would
like to use. For example, to compile an optimized binary for an Intel i9-9880H,
you might use:

```
OLLAMA_CUSTOM_CPU_DEFS="-DGGML_AVX=on -DGGML_AVX2=on -DGGML_F16C=on -DGGML_FMA=on" go generate ./...
go build .
```

#### Containerized Linux Build

If you have Docker available, you can build linux binaries with `./scripts/build_linux.sh` which has the CUDA and ROCm dependencies included. The resulting binary is placed in `./dist`

### Windows

Note: The Windows build for Ollama is still under development.

First, install required tools:

- MSVC toolchain - C/C++ and cmake as minimal requirements
- Go version 1.22 or higher
- MinGW (pick one variant) with GCC.
  - [MinGW-w64](https://www.mingw-w64.org/)
  - [MSYS2](https://www.msys2.org/)
- The `ThreadJob` Powershell module: `Install-Module -Name ThreadJob -Scope CurrentUser`

Then, build the `ollama` binary:

```powershell
$env:CGO_ENABLED="1"
go generate ./...
go build .
```

#### Windows CUDA (NVIDIA)

In addition to the common Windows development tools described above, install CUDA after installing MSVC.

- [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)


#### Windows ROCm (AMD Radeon)

In addition to the common Windows development tools described above, install AMDs HIP package after installing MSVC.

- [AMD HIP](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
- [Strawberry Perl](https://strawberryperl.com/)

Lastly, add `ninja.exe` included with MSVC to the system path (e.g. `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja`).
