# MLX engine

This is a small inference engine written in Go using [MLX](https://github.com/ml-explore/mlx), Apple's array framework for machine learning

## Goals

1. Implement multimodal runners: in a dedicated runner but eventually to be integrated into Ollama's primary runner.
2. Optimizing for image generation memory usage and output speed
3. (secondary): implement fast text model inference for gpt-oss, Llama.

## Prerequisites

**macOS:**

- macOS 14.0+ (Sonoma or later)
- Apple Silicon (M1/M2/M3)
- Xcode Command Line Tools

**Linux (building from source):**

- NVIDIA GPU (compute capability 7.0+)
- CUDA 12.0+ toolkit
- cuDNN

**Linux (prebuilt binaries):**

- NVIDIA GPU (compute capability 7.0+)
- NVIDIA driver 525+ (CUDA runtime libs are bundled)

**Both:**

- CMake 3.25+
- Go 1.21+

## Quick Start

### Build MLX

```bash
cmake -B build
cmake --build build --parallel
cmake --install build
```

This fetches MLX and mlx-c, builds them, and installs to `dist/`:

- `dist/lib/libmlxc.so` (or `.dylib`) - MLX C bindings
- `dist/lib/libmlx.a` - MLX static library
- `dist/include/` - Headers (mlx-c, CCCL for CUDA JIT)

To update MLX version, change `MLX_GIT_TAG` in `CMakeLists.txt` and rebuild.

### 2. Download a Model

Download Llama 3.1 8B (or any compatible model) in safetensors format:

```bash
mkdir -p ./weights

# Example using huggingface-cli
hf download meta-llama/Llama-3.1-8B --local-dir ./weights/Llama-3.1-8B
hf download openai/gpt-oss-20b --local-dir ./weights/gpt-oss-20b
```

### 3. Run Inference

```bash
# Build
go build ./cmd/engine

# Text generation
./engine -model ./weights/Llama-3.1-8B -prompt "Hello, world!" -max-tokens 250

# Qwen-Image 2512 (text-to-image)
./engine -qwen-image -model ./weights/Qwen-Image-2512 -prompt "A mountain landscape at sunset" \
  -width 1024 -height 1024 -steps 20 -seed 42 -output landscape.png

# Qwen-Image Edit (experimental) - 8 steps for speed, but model recommends 50
./engine -qwen-image-edit -model ./weights/Qwen-Image-Edit-2511 \
  -input-image input.png -prompt "Make it winter" -negative-prompt " " -cfg-scale 4.0 \
  -steps 8 -seed 42 -output edited.png
```

## Adding a Model

Use Claude Code with this repo. See `models/CLAUDE.md` for the full guide covering:

- Porting Python models to Go (forward pass, weight loading)
- Component testing with Python reference data
- Performance optimization

Reference implementations: `llama` (LLM), `qwen_image` (image generation), `qwen_image_edit` (image editing)

## Memory Management

MLX Python/C++ uses scope-based memory management - arrays are freed when they go out of scope. Go's garbage collector is non-deterministic, so we can't rely on finalizers to free GPU memory promptly.

Instead, arrays are automatically tracked and freed on `Eval()`:

```go
// All arrays are automatically tracked when created
x := mlx.Add(a, b)
y := mlx.Matmul(x, w)

// Eval frees non-kept arrays, evaluates outputs (auto-kept)
mlx.Eval(y)

// After copying to CPU, free the array
data := y.Data()
y.Free()
```

Key points:

- All created arrays are automatically tracked
- `mlx.Eval(outputs...)` frees non-kept arrays, evaluates outputs (outputs auto-kept)
- `mlx.Keep(arrays...)` marks arrays to survive multiple Eval cycles (for weights, caches)
- Call `.Free()` when done with an array

## Testing

### Running Tests

```bash
# Run all tests (tests skip if dependencies missing)
go test ./...

# Run specific model tests
go test ./models/qwen_image/...
```

### Model Weights

Tests require model weights in `./weights/<model-name>/`:

```
weights/
├── Qwen-Image-2512/      # Qwen image generation
│   ├── text_encoder/
│   ├── transformer/
│   ├── vae/
│   └── tokenizer/
├── Llama-3.1-8B/         # LLM
└── ...
```

Download models using `huggingface-cli`:

```bash
hf download ./weights/Qwen-Image-2512 --local-dir ./weights/Qwen-Image-2512
```
