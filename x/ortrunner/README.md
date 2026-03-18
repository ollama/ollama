# Ollama - NPU / ONNX Runtime Inference

Experimental support for running ONNX models on **NPU**, **GPU**, or **CPU** via
[ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) and
[DirectML](https://github.com/microsoft/DirectML).

This enables hardware-accelerated inference on Windows ARM64 devices with NPUs
(Qualcomm Snapdragon X Elite/Plus, Intel Core Ultra, AMD Ryzen AI, etc.)
without requiring vendor-specific SDKs.

## Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Windows 11 (ARM64 or x64) |
| **NPU Driver** | Device-specific — must expose a D3D12 adapter via DXCore |
| **Go** | 1.24+ with CGo enabled |
| **C Compiler** | MSVC (VS 2022) or LLVM/Clang targeting `*-windows-msvc` |
| **ONNX Model** | A GenAI-compatible ONNX model directory (see [Models](#models)) |

### NPU Driver Compatibility

NPU inference requires a driver that registers the NPU as a D3D12 adapter.
Verify your hardware is supported:

| Vendor | Chipset | NPU Driver | EP |
|---|---|---|---|
| Qualcomm | Snapdragon X Elite / Plus | Adreno GPU + Hexagon NPU (Windows Update) | `dml` or `qnn` |
| Intel | Core Ultra (Meteor Lake+) | Intel NPU Driver (Windows Update) | `dml` |
| AMD | Ryzen AI 300+ | AMD IPU Driver | `dml` |

To check if your NPU is accessible, open PowerShell:

```powershell
Get-CimInstance Win32_VideoController | Format-Table Name, Status -AutoSize
```

You should see your NPU listed with Status `OK`. If it only appears as a
display adapter with no D3D12 user-mode driver, NPU targeting will silently
fall back to GPU.

## Setup

### 1. Download DLLs

Place these three DLLs in `lib/ollama/ortgenai/` (or any directory you point
`OLLAMA_ORT_PATH` to):

| DLL | Source | Version Tested |
|---|---|---|
| `onnxruntime-genai.dll` | [onnxruntime-genai releases](https://github.com/microsoft/onnxruntime-genai/releases) | v0.12.1 ARM64 |
| `onnxruntime.dll` | NuGet: `Microsoft.ML.OnnxRuntime.DirectML` | v1.21.1 ARM64 |
| `DirectML.dll` | NuGet: `Microsoft.AI.DirectML` | v1.15.4 ARM64 |

**Important:** Match the architecture (ARM64 vs x64) to your OS. The
`onnxruntime.dll` **must** come from the DirectML NuGet package — the system
copy (often v1.17 from Edge/Office) is too old and will cause API version
errors.

### 2. Download an ONNX Model

You need a model in ORT GenAI format (a directory containing `genai_config.json`,
`model.onnx`, and tokenizer files).

```bash
# Example: Phi-3-mini 4K Instruct (DirectML, int4 quantized, ~2 GB)
git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
cd Phi-3-mini-4k-instruct-onnx
git lfs pull --include="directml/directml-int4-awq-block-128/*"
```

Other models that work with ORT GenAI + DirectML:
- `microsoft/Phi-4-mini-instruct-onnx`
- `microsoft/Phi-3.5-mini-instruct-onnx`
- Any [Hugging Face model with `onnx` + `genai` tags](https://huggingface.co/models?library=onnxruntime&sort=trending)

### 3. Build

On Windows ARM64 without MinGW GCC, use the `cgo-clang` wrapper scripts
(included in this repo) that filter GCC-specific flags for MSVC-targeted
Clang:

```bash
CC=c:/path/to/cgo-clang.exe \
CXX=c:/path/to/cgo-clang++.exe \
CGO_ENABLED=1 \
CGO_LDFLAGS="-ladvapi32" \
go build -o ollama.exe .
```

With MSVC or standard MinGW:

```bash
CGO_ENABLED=1 CGO_LDFLAGS="-ladvapi32" go build -o ollama.exe .
```

## Usage

### Quick Start (env var override)

The fastest way to test is using `OLLAMA_ONNX_MODEL` to route any model
request through the ORT GenAI runner:

```bash
OLLAMA_ORT_PATH=lib/ollama/ortgenai \
OLLAMA_ONNX_MODEL="c:/path/to/your/onnx-model-dir" \
./ollama.exe serve
```

Then in another terminal:

```bash
ollama run phi3
# Or any model name — it routes to the ONNX model regardless
```

### Targeting the NPU

Set `OLLAMA_ORT_DEVICE_TYPE=npu` to direct inference to the NPU via DirectML:

```bash
OLLAMA_ORT_PATH=lib/ollama/ortgenai \
OLLAMA_ONNX_MODEL="c:/path/to/your/onnx-model-dir" \
OLLAMA_ORT_DEVICE_TYPE=npu \
./ollama.exe serve
```

If the NPU is not available (no D3D12 driver), DirectML silently falls back to
GPU.

### Targeting a Specific Device

Use `OLLAMA_ORT_DEVICE_ID` to select a device by DXGI enumeration index:

```bash
OLLAMA_ORT_DEVICE_ID=1 ./ollama.exe serve
```

### Using QNN (Qualcomm Snapdragon only)

For Qualcomm devices with the QNN SDK installed, the QNN execution provider
can target the Hexagon HTP directly:

```bash
OLLAMA_ONNX_PROVIDER=qnn ./ollama.exe serve
```

This requires a QNN-specific ONNX model (not the same as DirectML models).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_ORT_PATH` | `lib/ollama/ortgenai` | Directory containing ORT GenAI DLLs |
| `OLLAMA_ONNX_MODEL` | *(none)* | Path to ONNX model directory; overrides normal model routing |
| `OLLAMA_ONNX_PROVIDER` | `dml` | Execution provider: `dml`, `qnn`, `cpu`, or any ORT EP name |
| `OLLAMA_ORT_DEVICE_TYPE` | *(none)* | Device filter: `npu` or `gpu` (DML only) |
| `OLLAMA_ORT_DEVICE_ID` | *(none)* | Explicit device index (overrides device type filter) |
| `OLLAMA_ONNX_NPU` | `0` | Set to `1` to switch to QNN provider |

## Architecture

```
ollama serve
  |
  |-- server/sched.go  (checks OLLAMA_ONNX_MODEL or model.IsONNX())
  |     |
  |     +-- ortrunner.NewClient(modelDir)
  |           |
  |           +-- spawns subprocess:
  |                 ollama runner --ortgenai-engine --model <dir> --port <port>
  |
  +-- x/ortrunner/server.go    HTTP server (health, completion, tokenize)
  +-- x/ortrunner/runner.go    Model loading + EP configuration
  +-- x/ortrunner/pipeline.go  Token-by-token generation loop
  +-- x/ortrunner/client.go    Subprocess client (implements llm.LlamaServer)
  +-- x/ortrunner/oga/         CGo bindings: dynamic loading of onnxruntime-genai.dll
```

The runner loads `onnxruntime-genai.dll` at runtime via `LoadLibraryA` +
`GetProcAddress` (no link-time dependency). A `SetDllDirectoryA` call ensures
the correct `onnxruntime.dll` is loaded from the same directory, avoiding
conflicts with system copies.

## Troubleshooting

### "The requested API version [23] is not available"

The wrong `onnxruntime.dll` is being loaded (system copy from Edge/Office).
Make sure `OLLAMA_ORT_PATH` points to the directory with **your** copy of
`onnxruntime.dll` (v1.21+).

### Model loads but output is garbage

Ensure the model variant matches the execution provider. DirectML models need
the DML provider; QNN models need the QNN provider. Using a DML-quantized
model with `OLLAMA_ONNX_PROVIDER=cpu` may produce incorrect results.

### "failed to load ORT GenAI dynamic library"

Check that all three DLLs are present and match your architecture:
```bash
ls lib/ollama/ortgenai/
# Should contain: onnxruntime-genai.dll  onnxruntime.dll  DirectML.dll
```

### NPU targeting has no effect

Run the `npu_access_probe` tool in `_tools/` to verify your NPU has a working
D3D12 driver. If `D3D12CreateDevice` fails for the NPU adapter, DirectML
cannot use it and will fall back to GPU.
