# Development

Install required tools:

- cmake version 3.24 or higher (only required for legacy C++ runner build)
- go version 1.22 or higher
- gcc version 11.4.0 or higher


## Transitional new Go llama Runner

The Ollama team is working on moving to a new Go based llama runner subprocess.  During a transition period, this new Go runner is "opt in" at build time, and requires using a different approach to build.  When you run `go generate ./...` you will build the C++ based runner.  To build the new Go runner, use `make` as described below. Once either the C++ or Go runners are built, simply run `go build .` as before.  After we complete the transition to use the Go server exclusively, both `make` and `go generate` will build the Go Runner.  The instructions below assume an "opt in" build of the new Go server.

### MacOS

```bash
brew install go gcc
```

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

Running make will compile several CPU runners which can run on different CPU families. At runtime, Ollama will auto-detect the best variation to load. 

To build your own custom CPU runner, set CUSTOM_CPU_FLAGS to a space delimited list of CPU flags. For example, to build a custom CPU runner with avx512, use the following:
```
make -C llama CUSTOM_CPU_FLAGS="avx avx2 avx512f avx512bw avx512vbmi avx512vnni avx512bf16"
go build .
```

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

### Custom CPU flags for GPU Runners

For both Windows and Linux GPU runners on x86, by deafult Ollama compiles with the "avx" CPU vector feature enabled.  This provides a good performance balance when loading large models that split across GPU and CPU with broad compatibility.  Some users may prefer no vector extensions (e.g. older Xeon/Celeron processors, or hypervisors that mask the vector features) while other users may prefer turning on many more vector extensions to further improve performance for split model loads.  Both scenarios can be accomplished by setting `GPU_RUNNER_CPU_FLAGS` during `make`

For example, to disable all vector flags for the GPU runners
```
make -C llama GPU_RUNNER_CPU_FLAGS="" -j 5
go build .
```

To enable a larger set of vector features
```
make -C llama GPU_RUNNER_CPU_FLAGS="avx avx2 avx512f avx512bw avx512vbmi avx512vnni avx512bf16"
go build .
```
