# Experimental Features 

## MLX Backend

We're working on a new experimental backend based on the [MLX project](https://github.com/ml-explore/mlx)

Support is currently limited to MacOS and Linux with CUDA GPUs.  We're looking to add support for Windows CUDA soon, and other GPU vendors.  To build:

```
cmake --preset MLX
cmake --build --preset MLX --parallel
cmake --install --component MLX
go build -tags mlx .
```

On linux, use the preset "MLX CUDA 13" or "MLX CUDA 12" to enable CUDA with the default Ollama NVIDIA GPU architectures enabled. 

## Image Generation

Based on the experimental MLX backend, we're working on adding imagegen support.  After running the cmake commands above:

```
go build -o imagegen ./x/imagegen/cmd/engine
```
