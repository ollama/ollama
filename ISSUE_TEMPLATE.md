**Issue Title:** Performance optimization: O(1) KV cache allocation using bitmap free block pool

## Problem

Current `findLocs()` in `kvcache/causal.go` uses O(n) linear search. For large, partially-filled caches:
- 50% full, 100K slots: ~66μs per allocation
- 90% full, 100K slots: ~130μs per allocation

This becomes a bottleneck in multi-tenant serving scenarios.

## Proposed Solution

Add bitmap-based free block pool (1 bit per cell):
- O(1) allocation via bitmap operations
- Negligible memory overhead (~125 bytes per 1K slots)

## Performance Impact

Benchmark with partially filled cache:

| Scenario | Original | Optimized | Speedup |
|----------|----------|-----------|---------|
| 50% full, 10K slots | 8.5μs | 70ns | **121x** |
| 90% full, 10K slots | 15.5μs | 79ns | **195x** |
| 50% full, 100K slots | 66μs | 146ns | **453x** |
| 90% full, 100K slots | 130μs | 106ns | **1224x** |

Trade-off: Empty cache ~2x slower (bitmap overhead), but this is negligible compared to the gains in realistic usage.

## Implementation

I have a working implementation with full test coverage. Files:
- `kvcache/cache_accelerated.go` - Optimized version
- `kvcache/cache_accelerated_test.go` - Unit tests (9/9 passing)
- `kvcache/bench_realistic_test.go` - Performance benchmarks

Would the maintainers be interested in reviewing this optimization? I can prepare a PR if this aligns with the project's goals.

## References

Similar approach used in vLLM's PagedAttention implementation.
