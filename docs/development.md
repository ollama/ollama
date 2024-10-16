# Development

> [!IMPORTANT]
> The `llm` package that loads and runs models is being updated to use a new [Go runner](#transition-to-go-runner): this should only impact a small set of PRs however it does change how the project is built.

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

#### Windows arm64

The default `Developer PowerShell for VS 2022` may default to x86 which is not what you want.  To ensure you get an arm64 development environment, start a plain PowerShell terminal and run:

```powershell
import-module 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\Microsoft.VisualStudio.DevShell.dll'
Enter-VsDevShell -Arch arm64 -vsinstallpath 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community' -skipautomaticlocation
```

You can confirm with `write-host $env:VSCMD_ARG_TGT_ARCH`

Follow the instructions at https://www.msys2.org/wiki/arm64/ to set up an arm64 msys2 environment.  Ollama requires gcc and mingw32-make to compile, which is not currently available on Windows arm64, but a gcc compatibility adapter is available via `mingw-w64-clang-aarch64-gcc-compat`. At a minimum you will need to install the following:

```
pacman -S mingw-w64-clang-aarch64-clang mingw-w64-clang-aarch64-gcc-compat mingw-w64-clang-aarch64-make make
```

You will need to ensure your PATH includes go, cmake, gcc and clang mingw32-make to build ollama from source. (typically `C:\msys64\clangarm64\bin\`)


## Transition to Go runner

The Ollama team is working on moving to a new Go based runner that loads and runs models in a subprocess to replace the previous code under `ext_server`. During this transition period, this new Go runner is "opt in" at build time, and requires using a different approach to build.

After the transition to use the Go server exclusively, both `make` and `go generate` will build the Go runner.

Install required tools:

- go version 1.22 or higher
- gcc version 11.4.0 or higher


### MacOS

[Download Go](https://go.dev/dl/)

Optionally enable debugging and more verbose logging:

```bash
# At build time
export CGO_CFLAGS="-g"

# At runtime
export OLLAMA_DEBUG=1
```

Get the required libraries and build the native LLM code:  (Adjust the job count based on your number of processors for a faster build)

```bash
make -C llama -j 5
```

Then build ollama:

```bash
go build .
```

Now you can run `ollama`:

```bash
./ollama
```

#### Xcode 15 warnings

If you are using Xcode newer than version 14, you may see a warning during `go build` about `ld: warning: ignoring duplicate libraries: '-lobjc'` due to Golang issue https://github.com/golang/go/issues/67799 which can be safely ignored.  You can suppress the warning with `export CGO_LDFLAGS="-Wl,-no_warn_duplicate_libraries"`

### Linux

#### Linux CUDA (NVIDIA)

_Your operating system distribution may already have packages for NVIDIA CUDA. Distro packages are often preferable, but instructions are distro-specific. Please consult distro-specific docs for dependencies if available!_

Install `make`, `gcc` and `golang` as well as [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
development and runtime packages.

Typically the build scripts will auto-detect CUDA, however, if your Linux distro
or installation approach uses unusual paths, you can specify the location by
specifying an environment variable `CUDA_LIB_DIR` to the location of the shared
libraries, and `CUDACXX` to the location of the nvcc compiler. You can customize
a set of target CUDA architectures by setting `CMAKE_CUDA_ARCHITECTURES` (e.g. "50;60;70")

Then generate dependencies:  (Adjust the job count based on your number of processors for a faster build)

```
make -C llama -j 5
```

Then build the binary:

```
go build .
```

#### Linux ROCm (AMD)

_Your operating system distribution may already have packages for AMD ROCm and CLBlast. Distro packages are often preferable, but instructions are distro-specific. Please consult distro-specific docs for dependencies if available!_

Install [CLBlast](https://github.com/CNugteren/CLBlast/blob/master/doc/installation.md) and [ROCm](https://rocm.docs.amd.com/en/latest/) development packages first, as well as `make`, `gcc`, and `golang`.

Typically the build scripts will auto-detect ROCm, however, if your Linux distro
or installation approach uses unusual paths, you can specify the location by
specifying an environment variable `ROCM_PATH` to the location of the ROCm
install (typically `/opt/rocm`), and `CLBlast_DIR` to the location of the
CLBlast install (typically `/usr/lib/cmake/CLBlast`). You can also customize
the AMD GPU targets by setting AMDGPU_TARGETS (e.g. `AMDGPU_TARGETS="gfx1101;gfx1102"`)

Then generate dependencies:  (Adjust the job count based on your number of processors for a faster build)

```
make -C llama -j 5
```

Then build the binary:

```
go build .
```

ROCm requires elevated privileges to access the GPU at runtime. On most distros you can add your user account to the `render` group, or run as root.

#### Advanced CPU Settings

By default, running `make` will compile a few different variations
of the LLM library based on common CPU families and vector math capabilities,
including a lowest-common-denominator which should run on almost any 64 bit CPU
somewhat slowly. At runtime, Ollama will auto-detect the optimal variation to
load. 

Custom CPU settings are not currently supported in the new Go server build but will be added back after we complete the transition.

#### Containerized Linux Build

If you have Docker available, you can build linux binaries with `OLLAMA_NEW_RUNNERS=1 ./scripts/build_linux.sh` which has the CUDA and ROCm dependencies included. The resulting binary is placed in `./dist`

### Windows

The following tools are required as a minimal development environment to build CPU inference support.

- Go version 1.22 or higher
  - https://go.dev/dl/
- Git
  - https://git-scm.com/download/win
- GCC and Make.  There are multiple options on how to go about installing these tools on Windows.  We have verified the following, but others may work as well:  
  - [MSYS2](https://www.msys2.org/)
    - After installing, from an MSYS2 terminal, run `pacman -S mingw-w64-ucrt-x86_64-gcc make` to install the required tools
  - Assuming you used the default install prefix for msys2 above, add `c:\msys64\ucrt64\bin` and `c:\msys64\usr\bin` to your environment variable `PATH` where you will perform the build steps below (e.g. system-wide, account-level, powershell, cmd, etc.)

Then, build the `ollama` binary:

```powershell
$env:CGO_ENABLED="1"
make -C llama -j 8
go build .
```

#### GPU Support

The GPU tools require the Microsoft native build tools.  To build either CUDA or ROCm, you must first install MSVC via Visual Studio:

- Make sure to select `Desktop development with C++` as a Workload during the Visual Studio install
- You must complete the Visual Studio install and run it once **BEFORE** installing CUDA or ROCm for the tools to properly register
- Add the location of the **64 bit (x64)** compiler (`cl.exe`) to your `PATH`
- Note: the default Developer Shell may configure the 32 bit (x86) compiler which will lead to build failures.  Ollama requires a 64 bit toolchain.

#### Windows CUDA (NVIDIA)

In addition to the common Windows development tools and MSVC described above:

- [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

#### Windows ROCm (AMD Radeon)

In addition to the common Windows development tools and MSVC described above:

- [AMD HIP](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)

#### Windows arm64

The default `Developer PowerShell for VS 2022` may default to x86 which is not what you want.  To ensure you get an arm64 development environment, start a plain PowerShell terminal and run:

```powershell
import-module 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\Microsoft.VisualStudio.DevShell.dll'
Enter-VsDevShell -Arch arm64 -vsinstallpath 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community' -skipautomaticlocation
```

You can confirm with `write-host $env:VSCMD_ARG_TGT_ARCH`

Follow the instructions at https://www.msys2.org/wiki/arm64/ to set up an arm64 msys2 environment.  Ollama requires gcc and mingw32-make to compile, which is not currently available on Windows arm64, but a gcc compatibility adapter is available via `mingw-w64-clang-aarch64-gcc-compat`. At a minimum you will need to install the following:

```
pacman -S mingw-w64-clang-aarch64-clang mingw-w64-clang-aarch64-gcc-compat mingw-w64-clang-aarch64-make make
```

You will need to ensure your PATH includes go, cmake, gcc and clang mingw32-make to build ollama from source. (typically `C:\msys64\clangarm64\bin\`)
