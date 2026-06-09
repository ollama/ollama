# MLX Engine

Experimental MLX backend for running models on Apple Silicon and CUDA.

## Build

```bash
go build -o engine ./x/imagegen/cmd/engine
```

## Text Generation

Text generation models are no longer supported by this engine.

## Image Generation

```bash
./engine -zimage -model /path/to/z-image -prompt "a cat" -output cat.png
```

Options:

- `-width`, `-height` - image dimensions (default 1024x1024)
- `-steps` - denoising steps (default 9)
- `-seed` - random seed (default 42)
