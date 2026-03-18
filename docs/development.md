# Desenvolvimento

Instale os pré-requisitos:

- [Go](https://go.dev/doc/install)
- Compilador C/C++, por exemplo, Clang no macOS, [TDM-GCC](https://github.com/jmeubank/tdm-gcc/releases/latest) (Windows amd64) ou [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (Windows arm64), GCC/Clang no Linux.

Em seguida, compile e execute o Ollama a partir do diretório raiz do repositório:

```shell
go run . serve
```

> [!NOTE]
> O Ollama inclui código nativo compilado com CGO. De tempos em tempos, essas estruturas de dados podem mudar e o CGO pode ficar fora de sincronia, resultando em falhas inesperadas. Você pode forçar uma compilação completa do código nativo executando `go clean -cache` primeiro.


## macOS (Apple Silicon)

O macOS Apple Silicon oferece suporte ao Metal, que já vem embutido no binário do Ollama. Nenhuma etapa adicional é necessária.

## macOS (Intel)

Instale os pré-requisitos:

- [CMake](https://cmake.org/download/) ou `brew install cmake`

Depois, configure e compile o projeto:

```shell
cmake -B build
cmake --build build
```

Por fim, execute o Ollama:

```shell
go run . serve
```

## Windows

Instale os pré-requisitos:

- [CMake](https://cmake.org/download/)
- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) incluindo o Native Desktop Workload
- (Opcional) Suporte a GPU AMD
    - [ROCm](https://rocm.docs.amd.com/en/latest/)
    - [Ninja](https://github.com/ninja-build/ninja/releases)
- (Opcional) Suporte a GPU NVIDIA
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network)
- (Opcional) Suporte a GPU VULKAN
    - [VULKAN SDK](https://vulkan.lunarg.com/sdk/home) - útil para GPUs AMD/Intel
- (Opcional) Suporte ao engine MLX
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)

Depois, configure e compile o projeto:

```shell
cmake -B build
cmake --build build --config Release
```

> Compilar para Vulkan requer a variável de ambiente VULKAN_SDK:
>
> PowerShell
> ```powershell
> $env:VULKAN_SDK="C:\VulkanSDK\<version>"
> ```
> CMD
> ```cmd
> set VULKAN_SDK=C:\VulkanSDK\<version>
> ```

> [!IMPORTANT]
> Compilar para ROCm requer flags adicionais:
> ```
> cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
> cmake --build build --config Release
> ```



Por fim, execute o Ollama:

```shell
go run . serve
```

## Windows (ARM)

O Windows ARM não oferece suporte a bibliotecas de aceleração adicionais no momento. Não use cmake; simplesmente `go run` ou `go build`.

## Linux

Instale os pré-requisitos:

- [CMake](https://cmake.org/download/) ou `sudo apt install cmake` ou `sudo dnf install cmake`
- (Opcional) Suporte a GPU AMD
    - [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
- (Opcional) Suporte a GPU NVIDIA
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads)
- (Opcional) Suporte a GPU VULKAN
    - [VULKAN SDK](https://vulkan.lunarg.com/sdk/home) - útil para GPUs AMD/Intel
    - Ou instale via gerenciador de pacotes: `sudo apt install vulkan-sdk` (Ubuntu/Debian) ou `sudo dnf install vulkan-sdk` (Fedora/CentOS)
- (Opcional) Suporte ao engine MLX
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)
    - OpenBLAS/LAPACK: `sudo apt install libopenblas-dev liblapack-dev liblapacke-dev` (Ubuntu/Debian)
> [!IMPORTANT]
> Certifique-se de que os pré-requisitos estejam no `PATH` antes de executar o CMake.


Depois, configure e compile o projeto:

```shell
cmake -B build
cmake --build build
```

Por fim, execute o Ollama:

```shell
go run . serve
```

## Engine MLX (Opcional)

O engine MLX permite executar modelos baseados em safetensor. É necessário compilar as bibliotecas compartilhadas [MLX](https://github.com/ml-explore/mlx) e [MLX-C](https://github.com/ml-explore/mlx-c) separadamente via CMake. No macOS, o MLX utiliza a biblioteca Metal para rodar na GPU; no Windows e no Linux, roda em GPUs NVIDIA via CUDA v13.

### macOS (Apple Silicon)

Requer a toolchain Metal. Instale o [Xcode](https://developer.apple.com/xcode/) primeiro e depois:

```shell
xcodebuild -downloadComponent MetalToolchain
```

Verifique se está instalado corretamente (deve imprimir "no input files"):

```shell
xcrun metal
```

Em seguida, compile:

```shell
cmake -B build --preset MLX
cmake --build build --preset MLX --parallel
cmake --install build --component MLX
```

> [!NOTE]
> Sem a toolchain Metal, o cmake concluirá silenciosamente com o Metal desativado. Verifique a saída do cmake por `Setting MLX_BUILD_METAL=OFF`, o que indica que a toolchain está ausente.

### Windows / Linux (CUDA)

Requer CUDA 13+ e [cuDNN](https://developer.nvidia.com/cudnn) 9+.

```shell
cmake -B build --preset "MLX CUDA 13"
cmake --build build --target mlx --target mlxc --config Release --parallel
cmake --install build --component MLX --strip
```

### Substituições locais de fonte do MLX

Para compilar usando um checkout local do MLX e/ou MLX-C (útil para desenvolvimento), defina variáveis de ambiente antes de executar o CMake:

```shell
export OLLAMA_MLX_SOURCE=/path/to/mlx
export OLLAMA_MLX_C_SOURCE=/path/to/mlx-c
```

Por exemplo, usando os scripts auxiliares com repositórios locais de mlx e mlx-c:
```shell
OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c ./scripts/build_linux.sh

OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c ./scripts/build_darwin.sh
```

```powershell
$env:OLLAMA_MLX_SOURCE="../mlx"
$env:OLLAMA_MLX_C_SOURCE="../mlx-c"
./scripts/build_darwin.ps1
```

## Docker

```shell
docker build .
```

### ROCm

```shell
docker build --build-arg FLAVOR=rocm .
```

## Executando testes

Para rodar os testes, use `go test`:

```shell
go test ./...
```

> NOTE: Em raras circunstâncias, pode ser necessário alterar um pacote usando o novo
> pacote "synctest" no go1.24.
>
> Se você não tiver o pacote "synctest" habilitado, não verá falhas de build ou
> teste resultantes das suas alterações (se houver) localmente, mas o CI irá falhar.
>
> Se vir falhas no CI, você pode continuar enviando alterações para ver se o build
> do CI passa, ou pode habilitar o pacote "synctest" localmente para ver as falhas
> antes de enviar.
>
> Para habilitar o pacote "synctest" para testes, execute o seguinte comando:
>
> ```shell
> GOEXPERIMENT=synctest go test ./...
> ```
>
> Se quiser habilitar o synctest para todos os comandos go, defina a variável de
> ambiente `GOEXPERIMENT` no seu perfil de shell ou usando:
>
> ```shell
> go env -w GOEXPERIMENT=synctest
> ```
>
> Isso habilitará o pacote "synctest" para todos os comandos go sem precisar
> defini-lo para todas as sessões do shell.
>
> O pacote synctest não é necessário para builds de produção.

## Detecção de bibliotecas

O Ollama procura bibliotecas de aceleração nos seguintes caminhos relativos ao executável `ollama`:

* `./lib/ollama` (Windows)
* `../lib/ollama` (Linux)
* `.` (macOS)
* `build/lib/ollama` (for development)

If the libraries are not found, Ollama will not run with any acceleration libraries.
