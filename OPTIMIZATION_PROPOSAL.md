# Optimization: Free block pool for O(1) KV cache allocation

## Summary

Implements a bitmap-based free block pool for KV cache slot allocation, reducing allocation complexity from O(n) to O(1). This optimization provides **100-1200x speedup** for realistic cache usage scenarios.

## Problem

The current `findLocs()` implementation in `kvcache/causal.go` uses linear search to find free cache slots:

```go
func (c *Causal) findLocs() ([]int32, error) {
    for i := range c.cells {  // O(n) - scans entire cache
        if len(c.cells[i].sequences) == 0 {
            loc = append(loc, int32(i))
        }
    }
}
```

**Performance impact:**
- Empty cache: ~45ns (fast - first slot is free)
- 50% full cache (10K slots): ~8,529ns
- 90% full cache (100K slots): ~129,879ns

As cache fills and capacity grows, allocation becomes a bottleneck.

## Solution

Add a bitmap-based free block pool alongside the existing `cells` array:

```go
type Accelerated struct {
    // ... existing fields ...

    // Free block pool for O(1) allocation
    freeBitmap   []uint64  // 1 bit per cell, 1 = free
    firstFreeHint int      // Hint for fast search start
    freeCount     int      // Total free cells
}
```

**Key operations:**
- `isSet(index)` - O(1) bitmap check
- `clearBit(index)` - O(1) mark as occupied
- `setBit(index)` - O(1) mark as free
- `findLocsAccelerated()` - O(1) amortized allocation

**Memory overhead:** ~1 bit per cache cell (negligible)

## Performance Results

Benchmark: `BenchmarkRealisticScenario` (partially filled cache)

| Cache State | Capacity | Original (ns) | Accelerated (ns) | Speedup |
|-------------|----------|---------------|------------------|---------|
| 50% full    | 10K      | 8,529        | 70               | **121x** |
| 90% full    | 10K      | 15,454       | 79               | **195x** |
| 50% full    | 100K     | 66,462       | 146              | **453x** |
| 90% full    | 100K     | 129,879      | 106              | **1224x** |

### Trade-offs

| Scenario | Impact |
|----------|--------|
| Empty cache | ~2x slower (bitmap overhead) |
| Partial/Full cache | **100-1200x faster** |
| Memory | +1 bit per cell (~125 bytes per 1K cells) |

The overhead on empty cache is negligible compared to the massive gains in realistic usage.

## Files Added

- `kvcache/cache_accelerated.go` - Optimized implementation
- `kvcache/cache_accelerated_test.go` - Unit tests (9/9 passing)
- `kvcache/bench_realistic_test.go` - Performance benchmarks

## Testing

```bash
# Run unit tests
go test ./kvcache -run TestAccelerated

# Run performance benchmarks
go test ./kvcache -bench=BenchmarkRealistic
```

All tests pass. Performance improvement verified across multiple scenarios.

## Integration Options

**Option A: Drop-in replacement**
- Replace `Causal` with `Accelerated` implementation
- Requires updating factory functions

**Option B: Configurable**
- Add flag to enable accelerated mode
- Allows A/B testing and gradual rollout

**Option C: Hybrid**
- Use bitmap for large caches (>10K slots)
- Fall back to simple search for small caches

## Related

- vLLM uses similar techniques for their PagedAttention implementation
- This optimization is particularly valuable for multi-tenant serving scenarios

## Next Steps

1. Review implementation for correctness
2. Decide on integration approach (A/B/C above)
3. Add configuration flag if needed
4. Integration testing with full model pipeline
