# PagedAttention KV Cache Implementation

## Summary

Implements vLLM-style PagedAttention KV cache for Ollama, using fixed-size blocks for efficient memory management and reduced fragmentation.

## Key Features

### Block-Based Allocation
- **Block Size**: 16 tokens per block (configurable, default from vLLM)
- **Physical Blocks**: Fixed-size memory blocks that can be allocated independently
- **Block Tables**: Each sequence maintains a mapping from logical to physical blocks
- **Non-Contiguous Storage**: Unlike Causal cache, blocks don't need to be contiguous

### Memory Management
- **Free Block Pool**: LIFO allocation for cache locality
- **Block Reference Counting**: Safe sharing and deallocation
- **Dynamic Allocation**: Blocks allocated on-demand as sequences grow
- **Efficient Reuse**: Freed blocks immediately available for reallocation

### API Compatibility
Implements the standard `Cache` interface:
- `Init()` - Initialize cache with capacity parameters
- `StartForward()` - Prepare for forward pass with block allocation
- `Get()` / `Put()` - Access KV tensors
- `CopyPrefix()` - Copy prefix between sequences
- `CanResume()` - Check if sequence can continue
- `Remove()` - Free token ranges

## Performance Characteristics

| Scenario | Paged Cache | Causal Cache |
|----------|-------------|--------------|
| Empty cache | Slightly slower (block overhead) |
| Partial cache | **O(1) block allocation** | O(n) scan |
| Fragmented | **No fragmentation** | External fragmentation |
| Memory overhead | +block table per seq | None |

## Memory Overhead

- **Block Table**: `O(num_sequences * avg_blocks_per_seq)` integers
- **Block Metadata**: `O(num_blocks)` entries for ref counts and mappings
- **Example**: 1000 sequences × 4 blocks × 4 bytes = ~16 KB

## Implementation Details

### Block Allocation Algorithm
```
1. Calculate blocks needed: ceil(tokens / block_size)
2. Allocate blocks from free pool
3. Update block table: seqID -> [physical_block_0, physical_block_1, ...]
4. On failure, free all allocated blocks and return error
```

### Location Calculation
```
logical_block = position / block_size
offset_in_block = position % block_size
physical_block = block_table[seqID][logical_block]
physical_location = physical_block * block_size + offset_in_block
```

### Block Deallocation
```
1. Calculate affected blocks: [begin_pos/block_size, end_pos/block_size)
2. Decrement reference counts
3. Free blocks with ref_count = 0
4. Update block table (remove freed blocks)
```

## Files Added

- `kvcache/paged.go` - Main PagedAttention implementation
- `kvcache/paged_test.go` - Unit tests (31 tests, all passing)
- `kvcache/paged_bench_test.go` - Performance benchmarks

## Testing

```bash
# Run unit tests
go test ./kvcache -run TestPaged -v

# Run benchmarks
go test ./kvcache -bench=BenchmarkPaged -benchmem
```

## Usage Example

```go
cache := NewPagedCache(shiftFn)
cache.Init(backend, ml.DTypeF16, maxSequences, capacity, maxBatch)

// Start forward pass - blocks allocated automatically
err := cache.StartForward(ctx, batch, false)

// Access KV data
keys, values, mask := cache.Get(ctx)

// Store KV data
cache.Put(ctx, keys, values)
```

## Future Enhancements

1. **Block Sharing**: Share identical blocks across sequences (for common prefixes)
2. **Eviction Policy**: LRU/LFU eviction when cache is full
3. **Block Size Tuning**: Dynamic block size based on workload
4. **CUDA Integration**: Specialized kernels for GPU block operations
5. **Prefetching**: Pre-allocate likely-to-be-needed blocks

## References

- vLLM: [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- vLLM GitHub: https://github.com/vllm-project/vllm
