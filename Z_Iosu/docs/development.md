# Development

Install prerequisites:

- [Go](https://go.dev/doc/install)
- C/C++ Compiler e.g. Clang on macOS, [TDM-GCC](https://github.com/jmeubank/tdm-gcc/releases/latest) (Windows amd64) or [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (Windows arm64), GCC/Clang on Linux.

Then build and run Ollama from the root directory of the repository:

```shell
go run . serve
```

## macOS (Apple Silicon)

macOS Apple Silicon supports Metal which is built-in to the Ollama binary. No additional steps are required.

## macOS (Intel)

Install prerequisites:

- [CMake](https://cmake.org/download/) or `brew install cmake`

Then, configure and build the project:

```shell
cmake -B build
cmake --build build
```

Lastly, run Ollama:

```shell
go run . serve
```

## Windows

Install prerequisites:

- [CMake](https://cmake.org/download/)
- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) including the Native Desktop Workload
- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/en/latest/)
    - [Ninja](https://github.com/ninja-build/ninja/releases)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network)

Then, configure and build the project:

```shell
cmake -B build
cmake --build build --config Release
```

> [!IMPORTANT]
> Building for ROCm requires additional flags:
> ```
> cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
> cmake --build build --config Release
> ```


Lastly, run Ollama:

```shell
go run . serve
```

## Windows (CUDA rápido con script)

Para acelerar un build local con GPU CUDA en Windows, usa el script incluido:

```powershell
# Ejemplo: arquitectura compute 89 (Ada Lovelace), instalar libs y binario Go
pwsh -File scripts/build-windows-cuda.ps1 -CudaArch "89" -Install -Verbose

# Múltiples arquitecturas (separadas por ; )
pwsh -File scripts/build-windows-cuda.ps1 -CudaArch "86;89" -Install

# Usar Ninja (si instalado) y omitir build Go para sólo libs
pwsh -File scripts/build-windows-cuda.ps1 -CudaArch "89" -UseNinja -SkipGo
```

Parámetros clave:
- `-CudaArch`: Lista de arquitecturas CUDA (equivalente a `CMAKE_CUDA_ARCHITECTURES`).
- `-UseNinja`: Usa Ninja como generador (más rápido si disponible).
- `-Install`: Ejecuta `cmake --install` para colocar libs en `dist/lib/ollama`.
- `-SkipGo`: No compila el binario Go (útil si sólo trabajas en backend C/C++).
- `-Reconfigure`: Borra y regenera el directorio `build`.

Una vez compilado:
```powershell
./ollama.exe serve
```

Si no se detecta la GPU:
```powershell
nvidia-smi
Get-ChildItem "$Env:CUDA_PATH\bin" | Select-String cudart
```

Asegúrate de que las DLLs copiadas están en `./lib/ollama`.

## Windows (ARM)

Windows ARM does not support additional acceleration libraries at this time.  Do not use cmake, simply `go run` or `go build`.

## Linux

Install prerequisites:

- [CMake](https://cmake.org/download/) or `sudo apt install cmake` or `sudo dnf install cmake`
- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads)

> [!IMPORTANT]
> Ensure prerequisites are in `PATH` before running CMake.


Then, configure and build the project:

```shell
cmake -B build
cmake --build build
```

Lastly, run Ollama:

```shell
go run . serve
```

## Docker

```shell
docker build .
```

### ROCm

```shell
docker build --build-arg FLAVOR=rocm .
```

## Running tests

To run tests, use `go test`:

```shell
go test ./...
```

> NOTE: In rare circumstances, you may need to change a package using the new
> "synctest" package in go1.24.
>
> If you do not have the "synctest" package enabled, you will not see build or
> test failures resulting from your change(s), if any, locally, but CI will
> break.
>
> If you see failures in CI, you can either keep pushing changes to see if the
> CI build passes, or you can enable the "synctest" package locally to see the
> failures before pushing.
>
> To enable the "synctest" package for testing, run the following command:
>
> ```shell
> GOEXPERIMENT=synctest go test ./...
> ```
>
> If you wish to enable synctest for all go commands, you can set the
> `GOEXPERIMENT` environment variable in your shell profile or by using:
>
> ```shell
> go env -w GOEXPERIMENT=synctest
> ```
>
> Which will enable the "synctest" package for all go commands without needing
> to set it for all shell sessions.
>
> The synctest package is not required for production builds.

## Library detection

Ollama looks for acceleration libraries in the following paths relative to the `ollama` executable:

* `./lib/ollama` (Windows)
* `../lib/ollama` (Linux)
* `.` (macOS)
* `build/lib/ollama` (for development)

If the libraries are not found, Ollama will not run with any acceleration libraries.
