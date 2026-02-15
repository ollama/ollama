# MLX Engine

Experimental MLX backend for running models on Apple Silicon and CUDA.

## Build

```bash
go build -tags mlx -o engine ./x/imagegen/cmd/engine
```

## Text Generation

```bash
./engine -model /path/to/model -prompt "Hello" -max-tokens 100
```

Options:

- `-temperature` - sampling temperature (default 0.7)
- `-top-p` - nucleus sampling (default 0.9)
- `-top-k` - top-k sampling (default 40)

Supports: Llama, Gemma3, GPT-OSS

## Image Generation

```bash
./engine -zimage -model /path/to/z-image -prompt "a cat" -output cat.png
```

Options:

- `-width`, `-height` - image dimensions (default 1024x1024)
- `-steps` - denoising steps (default 9)
- `-seed` - random seed (default 42)
