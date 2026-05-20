# Ollama para IBM Power9 (ppc64le) com GPU NVIDIA Tesla V100

Fork do Ollama oficial (v0.23.2) com suporte à arquitetura ppc64le para IBM AC922 com CUDA.

**Branch:** Todo o trabalho está na branch `ollama-ppc64le`. As demais branches são do repositório oficial do Ollama e podem ser ignoradas.

## Binário pré-compilado

Se você não quiser compilar, baixe o binário diretamente na página de releases:
https://github.com/llm-pt-ibm/ollama-ppc64le/releases/tag/v0.23.2-ppc64le-power9

```bash
# baixe o binário 
wget https://github.com/llm-pt-ibm/ollama-ppc64le/releases/download/v0.23.2-ppc64le-power9/ollama-ppc64le

# dê permissão de execução
chmod +x ollama-ppc64le
```

## Ambiente utilizado

**Hardware**:
- Arquitetura *ppc64le*;
- RAM: mínimo recomendado de ~64GB;
- GPU: NVIDIA Tesla V100;
- *Driver* NVIDIA: 535.54.03;
- CUDA: versão 12.2.

**Sistema Operacional:** Alma Linux 8.10 (*ppc64le*), binário compatível com *Red Hat Enterprise Linux (RHEL)* 8.9/8.10.

## Como compilar

### Passo 1 — Dependências

Para compilar o Ollama na POWER9, são necessárias as seguintes dependências com as versões adequadas:

Go: 1.26.0
GCC: 11.2.1 (via gcc-toolset-11)
CMake: >= 3.24

Instale as dependências:
```bash
sudo dnf update -y
sudo dnf install -y wget git tar make gcc gcc-c++ cmake gcc-toolset-11
```

Instale o Go 1.26.0:

```bash
wget https://go.dev/dl/go1.26.0.linux-ppc64le.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.26.0.linux-ppc64le.tar.gz
export PATH=/usr/local/go/bin:$PATH
```

Caso o CMake disponível seja anterior a 3.24, instale manualmente:

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.26.5/cmake-3.26.5.tar.gz
tar -xzf cmake-3.26.5.tar.gz
cd cmake-3.26.5
./bootstrap
make -j$(nproc)
sudo make install
cd ..
```

### Passo 2 — Ambiente de compilação

Esses exports precisam ser feitos toda vez que abrir um novo terminal:

```bash
export PATH=/usr/local/cuda-12.2/bin:/opt/rh/gcc-toolset-11/root/usr/bin:/usr/local/go/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:/opt/rh/gcc-toolset-11/root/usr/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda-12.2/bin/nvcc
```

**Importante:** No AlmaLinux 8, o `gcc-toolset` não é ativado automaticamente. Se estiver usando conda, o comando `scl enable` conflita com o ambiente, então use o `export` manual acima. Sem o ambiente virtual conda, você pode usar `scl enable gcc-toolset-11 bash`.

Verifique as versões:

```bash
gcc --version    # deve mostrar 11.2.1
nvcc --version   # deve mostrar CUDA 12.2
cmake --version  # deve mostrar 3.24
go version       # deve mostrar 1.26.0
```

### Passo 3 — Clonar o repositório

```bash
git clone https://github.com/llm-pt-ibm/ollama-ppc64le.git ollama-gpu
cd ollama-gpu
git checkout ollama-ppc64le
```

### Passo 4 — Aplicar fixes para ppc64le

```bash
mkdir -p ml/backend/ggml/ggml/src/ggml-cpu/arch/powerpc

curl -L -o ml/backend/ggml/ggml/src/ggml-cpu/arch/powerpc/cpu-feats.cpp \
  https://raw.githubusercontent.com/ggml-org/ggml/master/src/ggml-cpu/arch/powerpc/cpu-feats.cpp

curl -L -o ml/backend/ggml/ggml/src/ggml-cpu/arch/powerpc/quants.c \
  https://raw.githubusercontent.com/ggml-org/ggml/master/src/ggml-cpu/arch/powerpc/quants.c

sed -i '/ggml_add_cpu_backend_variant(power10/d' ml/backend/ggml/ggml/src/CMakeLists.txt
sed -i '/ggml_add_cpu_backend_variant(power11/d' ml/backend/ggml/ggml/src/CMakeLists.txt

cat > ml/backend/ggml/ggml/src/ggml-cpu/arch/powerpc/powerpc.go << 'EOF'
package powerpc
// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/../.. -I${SRCDIR}/../../.. -I${SRCDIR}/../../../../include
import "C"
EOF

cat > ml/backend/ggml/ggml/src/ggml-cpu/cpu_ppc64le.go << 'EOF'
package cpu
import _ "github.com/ollama/ollama/ml/backend/ggml/ggml/src/ggml-cpu/arch/powerpc"
EOF

sed -i '/add_subdirectory.*ggml-cuda/s|^|#|' CMakeLists.txt
```

### Passo 5 — Compilar com CMake

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release --parallel 8

sudo cmake --install build --prefix /usr/local
```

O `CUDA_ARCHITECTURES=70` corresponde à Tesla V100 (arquitetura Volta, `sm_70`). A compilação CUDA demora alguns minutos.

### Passo 6 — Compilar o binário Go

```bash
go build -trimpath \
  -ldflags="-X=github.com/ollama/ollama/version.Version=0.23.2 -extldflags=-lstdc++fs" \
  -o ollama .
```

Verifique: `./ollama --version` deve mostrar `ollama version is 0.23.2`.

## Como usar

```bash
# Suba o servidor
./ollama serve &
sleep 3

# Confirme que a GPU foi detectada nos logs:
# inference compute ... library=CUDA compute=7.0 ... description="Tesla V100-SXM2-16GB"

# Baixe um modelo
./ollama pull gemma3:12b

# Execute a inferência
./ollama run gemma3:12b "olá!"
```

## Solução de problemas

**GPU não detectada (`library=cpu` nos logs):**
O Ollama não encontrou as libs CUDA. Verifique se o `LD_LIBRARY_PATH` está correto antes de subir o servidor:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:/opt/rh/gcc-toolset-11/root/usr/lib64:$LD_LIBRARY_PATH
./ollama serve
```

**Erro `pull model manifest: 412`:**
O registry rejeitou o cliente por versão incompatível. Certifique-se de compilar o binário Go com o flag `-ldflags="-X=github.com/ollama/ollama/version.Version=0.23.2"`.

**Erro `go.mod requires go >= 1.26.0`:**
A versão do Go instalada é antiga. Atualize conforme o Passo 1.

**GPUs ocupadas por outro processo:**
```bash
nvidia-smi
fuser /dev/nvidia*
```
Use `CUDA_VISIBLE_DEVICES=N` para selecionar uma GPU específica.
