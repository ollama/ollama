# Experimental Features

## MLX Backend

We're working on a new experimental backend based on the [MLX project](https://github.com/ml-explore/mlx)

Support is currently limited to MacOS and Linux with CUDA GPUs. We're looking to add support for Windows CUDA soon, and other GPU vendors.

### Building ollama-mlx

The `ollama-mlx` binary is a separate build of Ollama with MLX support enabled. This enables experimental features like image generation.

#### macOS (Apple Silicon and Intel)

```bash
# Build MLX backend libraries
cmake --preset MLX
cmake --build --preset MLX --parallel
cmake --install build --component MLX

# Build ollama-mlx binary
go build -tags mlx -o ollama-mlx .
```

#### Linux (CUDA)

On Linux, use the preset "MLX CUDA 13" or "MLX CUDA 12" to enable CUDA with the default Ollama NVIDIA GPU architectures enabled:

```bash
# Build MLX backend libraries with CUDA support
cmake --preset 'MLX CUDA 13'
cmake --build --preset 'MLX CUDA 13' --parallel
cmake --install build --component MLX

# Build ollama-mlx binary
CGO_CFLAGS="-O3 -I$(pwd)/build/_deps/mlx-c-src" \
CGO_LDFLAGS="-L$(pwd)/build/lib/ollama -lmlxc -lmlx" \
go build -tags mlx -o ollama-mlx .
```

#### Using build scripts

The build scripts automatically create the `ollama-mlx` binary:

- **macOS**: `./scripts/build_darwin.sh` produces `dist/darwin/ollama-mlx`
- **Linux**: `./scripts/build_linux.sh` produces `ollama-mlx` in the output archives

## Image Generation

Image generation is built into the `ollama-mlx` binary. Run `ollama-mlx serve` to start the server with image generation support enabled.
