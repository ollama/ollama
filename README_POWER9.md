# Ollama for IBM Power9 (ppc64le) with NVIDIA Tesla V100 GPU

This is an Ollama fork (v0.23.2) with support for the ppc64le architecture on IBM AC922 systems using CUDA.

This project is part of the `IBM - MultiArq` initiative, a Computer Science course project at UFCG in partnership with IBM.

**Branch:** All work is on the `ollama-ppc64le` branch. Other branches are from the official Ollama repository and can be ignored.

## Pre-built binary

If you do not want to build from source, download the binary directly from the releases page:
https://github.com/llm-pt-ibm/ollama-ppc64le/releases/tag/v0.23.2-ppc64le-power9

```bash
# download the binary
wget https://github.com/llm-pt-ibm/ollama-ppc64le/releases/download/v0.23.2-ppc64le-power9/ollama-ppc64le

# make it executable
chmod +x ollama-ppc64le
```

## Environment used

**Hardware**:
- Architecture: *ppc64le*
- RAM: recommended minimum ~64GB
- GPU: NVIDIA Tesla V100
- NVIDIA driver: 535.54.03
- CUDA: version 12.2

**Operating System:** AlmaLinux 8.10 (*ppc64le*), binary compatible with *Red Hat Enterprise Linux (RHEL)* 8.9/8.10.

## How to build

### Step 1 — Dependencies

To build Ollama on POWER9, the following dependencies are required with these versions:

- Go: 1.26.0
- GCC: 11.2.1 (via gcc-toolset-11)
- CMake: >= 3.24

Install dependencies:
```bash
sudo dnf update -y
sudo dnf install -y wget git tar make gcc gcc-c++ cmake gcc-toolset-11
```

Install Go 1.26.0:

```bash
wget https://go.dev/dl/go1.26.0.linux-ppc64le.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.26.0.linux-ppc64le.tar.gz
export PATH=/usr/local/go/bin:$PATH
```

If the available CMake version is older than 3.24, install it manually:

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.26.5/cmake-3.26.5.tar.gz
tar -xzf cmake-3.26.5.tar.gz
cd cmake-3.26.5
./bootstrap
make -j$(nproc)
sudo make install
cd ..
```

### Step 2 — Build environment

These exports must be set every time you open a new terminal:

```bash
export PATH=/usr/local/cuda-12.2/bin:/opt/rh/gcc-toolset-11/root/usr/bin:/usr/local/go/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:/opt/rh/gcc-toolset-11/root/usr/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda-12.2/bin/nvcc
```

**Important:** On AlmaLinux 8, the `gcc-toolset` is not activated automatically. If you are using conda, `scl enable` conflicts with the environment, so use the manual `export` commands above. Without the conda environment, you can use `scl enable gcc-toolset-11 bash`.

Verify versions:

```bash
gcc --version    # should show 11.2.1
nvcc --version   # should show CUDA 12.2
cmake --version  # should show 3.24+
go version       # should show 1.26.0
```

### Step 3 — Clone the repository

```bash
git clone https://github.com/llm-pt-ibm/ollama-ppc64le.git ollama-gpu
cd ollama-gpu
git checkout ollama-ppc64le
```

### Step 4 — Apply ppc64le fixes

These fixes were required to successfully compile Ollama on the ppc64le architecture.

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

### Step 5 — Build with CMake

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release --parallel 8

sudo cmake --install build --prefix /usr/local
```

The `CUDA_ARCHITECTURES=70` value corresponds to the Tesla V100 (Volta architecture, `sm_70`). CUDA compilation may take several minutes.

### Step 6 — Build the Go binary

```bash
go build -trimpath \
  -ldflags="-X=github.com/ollama/ollama/version.Version=0.23.2 -extldflags=-lstdc++fs" \
  -o ollama .
```

Verify: `./ollama --version` should display `ollama version is 0.23.2`.

## How to use

```bash
# Start the server
./ollama serve &
sleep 3

# Confirm the GPU was detected in the logs:
# inference compute ... library=CUDA compute=7.0 ... description="Tesla V100-SXM2-16GB"

# Pull a model (replace with your preferred model)
./ollama pull <model-name>

# Run inference with the same model to verify it is working
./ollama run <model-name> "hello!"
```

## Troubleshooting

**CUDA not detected:**
If GPUs are not detected, check the NVIDIA drivers and CUDA libraries:

```bash
nvidia-smi

echo $LD_LIBRARY_PATH
ldconfig -p | grep cuda
```

Then enable Ollama debug logging and start the server:

```bash
export OLLAMA_DEBUG=1
./ollama serve
```

**GPU not detected (`library=cpu` in logs):**
Ollama did not find the CUDA libraries. Make sure `LD_LIBRARY_PATH` is set correctly before starting the server:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:/opt/rh/gcc-toolset-11/root/usr/lib64:$LD_LIBRARY_PATH
./ollama serve
```

**Error `pull model manifest: 412`:**
The registry rejected the client due to an incompatible version. Make sure you build the Go binary with the flag `-ldflags="-X=github.com/ollama/ollama/version.Version=0.23.2"`.

**Error `go.mod requires go >= 1.26.0`:**
The installed Go version is too old. Update it as described in Step 1.

**GPUs occupied by another process:**
```bash
nvidia-smi
fuser /dev/nvidia*
```
Use `CUDA_VISIBLE_DEVICES=N` to select a specific GPU.
