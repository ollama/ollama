# Development

Install required tools:

- go version 1.22 or higher
- OS specific C/C++ compiler (see below)
- GNU Make


## Overview

Ollama uses a mix of Go and C/C++ code to interface with GPUs.  The C/C++ code is compiled with both CGO and GPU library specific compilers.  A set of GNU Makefiles are used to compile the project.  GPU Libraries are auto-detected based on the typical environment variables used by the respective libraries, but can be overridden if necessary.  The default make target will build the runners and primary Go Ollama application that will run within the repo directory.  Throughout the examples below `-j 5` is suggested for 5 parallel jobs to speed up the build.  You can adjust the job count based on your CPU Core count to reduce build times.  If you want to relocate the built binaries, use the `dist` target and recursively copy the files in `./dist/$OS-$ARCH/` to your desired location. To learn more about the other make targets use `make help`

Once you have built the GPU/CPU runners, you can compile the main application with `go build .` 

### MacOS

[Download Go](https://go.dev/dl/)

```bash
make -j 5
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

Typically the makefile will auto-detect CUDA, however, if your Linux distro
or installation approach uses alternative paths, you can specify the location by
overriding `CUDA_PATH` to the location of the CUDA toolkit. You can customize
a set of target CUDA architectures by setting `CUDA_ARCHITECTURES` (e.g. `CUDA_ARCHITECTURES=50;60;70`)

```
make -j 5
```

If both v11 and v12 tookkits are detected, runners for both major versions will be built by default.  You can build just v12 with `make cuda_v12`

#### Older Linux CUDA (NVIDIA)

To support older GPUs with Compute Capability 3.5 or 3.7, you will need to use an older version of the Driver from [Unix Driver Archive](https://www.nvidia.com/en-us/drivers/unix/) (tested with 470) and [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) (tested with cuda V11).  When you build Ollama, you will need to set two make variable to adjust the minimum compute capability Ollama supports via `make -j 5 CUDA_ARCHITECTURES="35;37;50;52" EXTRA_GOLDFLAGS="\"-X=github.com/ollama/ollama/discover.CudaComputeMajorMin=3\" \"-X=github.com/ollama/ollama/discover.CudaComputeMinorMin=5\""`.  To find the Compute Capability of your older GPU, refer to [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).

#### Linux ROCm (AMD)

_Your operating system distribution may already have packages for AMD ROCm. Distro packages are often preferable, but instructions are distro-specific. Please consult distro-specific docs for dependencies if available!_

Install [ROCm](https://rocm.docs.amd.com/en/latest/) development packages first, as well as `make`, `gcc`, and `golang`.

Typically the build scripts will auto-detect ROCm, however, if your Linux distro
or installation approach uses unusual paths, you can specify the location by
specifying an environment variable `HIP_PATH` to the location of the ROCm
install (typically `/opt/rocm`). You can also customize
the AMD GPU targets by setting HIP_ARCHS (e.g. `HIP_ARCHS=gfx1101;gfx1102`)

```
make -j 5
```

ROCm requires elevated privileges to access the GPU at runtime. On most distros you can add your user account to the `render` group, or run as root.

#### Containerized Linux Build

If you have Docker and buildx available, you can build linux binaries with `./scripts/build_linux.sh` which has the CUDA and ROCm dependencies included. The resulting artifacts are placed in `./dist`  and by default the script builds both arm64 and amd64 binaries.  If you want to build only amd64, you can build with `PLATFORM=linux/amd64 ./scripts/build_linux.sh`

### Windows

The following tools are required as a minimal development environment to build CPU inference support.

- Go version 1.22 or higher
  - https://go.dev/dl/
- Git
  - https://git-scm.com/download/win
- clang with gcc compat and Make.  There are multiple options on how to go about installing these tools on Windows.  We have verified the following, but others may work as well:  
  - [MSYS2](https://www.msys2.org/)
    - After installing, from an MSYS2 terminal, run `pacman -S mingw-w64-clang-x86_64-gcc-compat mingw-w64-clang-x86_64-clang make` to install the required tools
  - Assuming you used the default install prefix for msys2 above, add `C:\msys64\clang64\bin` and `c:\msys64\usr\bin` to your environment variable `PATH` where you will perform the build steps below (e.g. system-wide, account-level, powershell, cmd, etc.)

> [!NOTE]  
> Due to bugs in the GCC C++ library for unicode support, Ollama should be built with clang on windows.

```
make -j 5
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


## Advanced CPU Vector Settings

On x86, running `make` will compile several CPU runners which can run on different CPU families. At runtime, Ollama will auto-detect the best variation to load.  If GPU libraries are present at build time, Ollama also compiles GPU runners with the `AVX` CPU vector feature enabled.  This provides a good performance balance when loading large models that split across GPU and CPU with broad compatibility.  Some users may prefer no vector extensions (e.g. older Xeon/Celeron processors, or hypervisors that mask the vector features) while other users may prefer turning on many more vector extensions to further improve performance for split model loads.

To customize the set of CPU vector features enabled for a CPU runner and all GPU runners, use CUSTOM_CPU_FLAGS during the build.

To build without any vector flags:

```
make CUSTOM_CPU_FLAGS=""
```

To build with both AVX and AVX2:
```
make CUSTOM_CPU_FLAGS=avx,avx2
```

To build with AVX512 features turned on:

```
make CUSTOM_CPU_FLAGS=avx,avx2,avx512,avx512vbmi,avx512vnni,avx512bf16
```

> [!NOTE]  
> If you are experimenting with different flags, make sure to do a `make clean` between each change to ensure everything is rebuilt with the new compiler flags
