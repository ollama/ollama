# Memory Optimizations

This directory contains memory optimization implementations for transformer models, including gradient checkpointing and KV cache compression.

## Background

The optimizations implemented here are based on research in computational pebbling and memory-efficient deep learning:

### Gradient Checkpointing (Pebbling)
- **sqrt(n) checkpointing strategy**: Based on optimal pebbling algorithms that achieve O(sqrt(n)) space complexity
- **Key papers**:
  - Griewank, A. and Walther, A. (2000). "Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation"
  - Chen, T. et al. (2016). "Training Deep Nets with Sublinear Memory Cost"
  - Kumar, M. et al. (2019). "Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization"

### Multi-Head Latent Attention (MLA) Compression
- **28:1 compression ratio**: Efficient KV cache compression for attention mechanisms
- **Key papers**:
  - Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
  - DeepSeek-AI (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"

## Implementation

### Core Components

- `verified_optimizations.go`: Main optimization algorithms
- `multi_vendor_gpu.go`: Cross-platform GPU backend (NVIDIA/AMD/Intel/Apple)
- Property-based testing suite with QuickCheck-style generators

### Key Algorithms

1. **Checkpoint Memory Estimation**: `sqrt(layers) + 1` checkpoints
2. **MLA Compression**: Fixed 28:1 ratio for KV cache
3. **GPU Device Scoring**: Memory-weighted performance estimation

## Usage

```go
// Create optimizer with both optimizations enabled
optimizer := NewVerifiedMemoryOptimizer(true, true)

// Estimate memory usage
checkpoints := optimizer.CheckpointMemoryEstimate(64) // 9 checkpoints for 64 layers
compressed := optimizer.MLACompressionEstimate(1500)  // 53 units from 1500 KV cache

// Get optimization statistics
stats := optimizer.GetOptimizationStats(64, 1500)
efficiency := stats["memory_efficiency"].(float64) // ~96% savings
```

## Testing

The implementation includes comprehensive property-based testing:

- **Checkpoint properties**: Memory reduction guarantees for layers ≥ 4
- **Compression properties**: 28:1 ratio maintenance and monotonicity
- **GPU properties**: Deterministic scoring and vendor detection
- **Integration properties**: Combined optimization effectiveness

Run tests with:
```bash
go test ./llm -run TestQuickCheck -v
```

## Performance Impact

For large models like DeepSeek V3:
- **Original memory**: 1564 units (64 layers + 1500 KV cache)
- **Optimized memory**: 62 units (9 checkpoints + 53 compressed KV)
- **Total savings**: 96% memory reduction

This enables training and inference of much larger models on the same hardware.