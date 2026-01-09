# imagegen

This is a package that uses MLX to run image generation models, ahead of being integrated into Ollama's primary runner.
in `CMakeLists.txt` and rebuild.

### 1. Download a Model

Download Llama 3.1 8B (or any compatible model) in safetensors format:

```bash
mkdir -p ./weights

# Example using huggingface-cli
hf download meta-llama/Llama-3.1-8B --local-dir ./weights/Llama-3.1-8B
hf download openai/gpt-oss-20b --local-dir ./weights/gpt-oss-20b
```

### 2. Run Inference

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
