package kvcache

import (
	"math"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// mockBackend embeds ml.Backend and only overrides needed methods.
type mockBackend struct {
	ml.Backend
}

func (b *mockBackend) NewContext() ml.Context {
	return &mockContext{}
}

func (b *mockBackend) NewContextSize(int) ml.Context {
	return &mockContext{}
}

func (b *mockBackend) CacheConfig() ml.CacheConfig {
	return ml.CacheConfig{
		CachePadding: 1,
		MaskDType:    ml.DTypeF32,
	}
}

// mockContext embeds ml.Context and only overrides needed methods.
type mockContext struct {
	ml.Context
}

func (c *mockContext) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	total := 1
	for _, s := range shape {
		total *= s
	}
	return &testTensor{dtype: dtype, elementSize: 4, data: make([]float32, total), shape: shape}
}

func (c *mockContext) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	return c.Empty(dtype, shape...)
}

func (c *mockContext) FromFloats(s []float32, shape ...int) ml.Tensor {
	t := c.Empty(ml.DTypeF32, shape...).(*testTensor)
	copy(t.data, s)
	return t
}

func (c *mockContext) FromInts(s []int32, shape ...int) ml.Tensor {
	f := make([]float32, len(s))
	for i := range f {
		f[i] = float32(s[i])
	}
	out := c.FromFloats(f, shape...).(*testTensor)
	out.dtype = ml.DTypeI32
	return out
}

func (c *mockContext) Arange(start, stop, step float32, dtype ml.DType) ml.Tensor {
	s := make([]float32, 0, int((stop-start)/step))
	for i := start; i < stop; i += step {
		s = append(s, i)
	}
	out := c.FromFloats(s, len(s)).(*testTensor)
	out.dtype = dtype
	return out
}

func (c *mockContext) Input() ml.Context    { return c }
func (c *mockContext) Layer(int) ml.Context { return c }
func (c *mockContext) Forward(...ml.Tensor) ml.Context {
	return c
}
func (c *mockContext) Compute(...ml.Tensor) {}
func (c *mockContext) ComputeWithNotify(func(), ...ml.Tensor) {}
func (c *mockContext) Reserve()                                     {}
func (c *mockContext) MaxGraphNodes() int                           { return 10 }
func (c *mockContext) Close() {}

// BenchmarkPagedCache_BlockAllocation benchmarks block allocation speed.
func BenchmarkPagedCache_BlockAllocation(b *testing.B) {
	cache := NewPagedCacheWithConfig(nil, 16, 10000)
	cache.Init(&mockBackend{}, ml.DTypeF32, 100, 256, 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seqID := i % 100
		cache.allocateBlocksForSequence(seqID, 32)
	}
}

// BenchmarkPagedCache_BlockTableLookup benchmarks block table lookup speed.
func BenchmarkPagedCache_BlockTableLookup(b *testing.B) {
	cache := NewPagedCacheWithConfig(nil, 16, 10000)
	cache.Init(&mockBackend{}, ml.DTypeF32, 100, 256, 100)

	// Pre-allocate blocks for sequences
	for i := 0; i < 100; i++ {
		cache.allocateBlocksForSequence(i, 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seqID := i % 100
		pos := int32(i % 256)
		logicalBlock := int(pos) / cache.blockSize
		if blocks, ok := cache.blockTables[seqID]; ok && logicalBlock < len(blocks) {
			_ = blocks[logicalBlock]
		}
	}
}

// BenchmarkPagedCache_StartForward benchmarks StartForward with varying batch sizes.
func BenchmarkPagedCache_StartForward(b *testing.B) {
	cache := NewPagedCacheWithConfig(nil, 16, 10000)
	cache.Init(&mockBackend{}, ml.DTypeF32, 100, 2048, 100)

	ctx := &mockContext{}

	batches := []struct {
		name      string
		seqs      int
		positions []int32
	}{
		{"small_batch_4_seqs", 4, []int32{0, 1, 2, 3}},
		{"medium_batch_16_seqs", 16, []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
		{"large_batch_64_seqs", 64, func() []int32 {
			pos := make([]int32, 64)
			for i := range pos {
				pos[i] = int32(i)
			}
			return pos
		}()},
	}

	for _, bb := range batches {
		b.Run(bb.name, func(b *testing.B) {
			seqs := make([]int, bb.seqs)
			for i := range seqs {
				seqs[i] = i
			}

			batch := input.Batch{
				Sequences: seqs,
				Positions: bb.positions,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cache.StartForward(ctx, batch, true)
			}
		})
	}
}

// BenchmarkPagedCache_CanResume benchmarks CanResume lookup speed.
func BenchmarkPagedCache_CanResume(b *testing.B) {
	cache := NewPagedCacheWithConfig(nil, 16, 10000)
	cache.Init(&mockBackend{}, ml.DTypeF32, 100, 256, 100)

	// Pre-allocate blocks
	for i := 0; i < 100; i++ {
		cache.allocateBlocksForSequence(i, 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seqID := i % 100
		pos := int32(i % 256)
		_ = cache.CanResume(seqID, pos)
	}
}

// BenchmarkPagedCache_GetStats benchmarks statistics collection.
func BenchmarkPagedCache_GetStats(b *testing.B) {
	cache := NewPagedCacheWithConfig(nil, 16, 10000)
	cache.Init(&mockBackend{}, ml.DTypeF32, 100, 256, 100)

	// Pre-allocate blocks
	for i := 0; i < 100; i++ {
		cache.allocateBlocksForSequence(i, 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cache.GetStats()
	}
}

// BenchmarkPagedCache_Remove benchmarks block removal speed.
func BenchmarkPagedCache_Remove(b *testing.B) {
	cache := NewPagedCacheWithConfig(nil, 16, 10000)
	cache.Init(&mockBackend{}, ml.DTypeF32, 100, 256, 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seqID := i % 100
		// Allocate
		cache.allocateBlocksForSequence(seqID, 64)
		// Remove middle portion
		cache.Remove(seqID, 16, 48)
	}
}

// BenchmarkPagedCache_CopyPrefix benchmarks prefix copying speed.
func BenchmarkPagedCache_CopyPrefix(b *testing.B) {
	cache := NewPagedCacheWithConfig(nil, 16, 10000)
	cache.Init(&mockBackend{}, ml.DTypeF32, 100, 256, 100)

	// Pre-allocate source sequences
	for i := 0; i < 100; i++ {
		cache.allocateBlocksForSequence(i, 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srcSeq := i % 100
		dstSeq := (i + 100) % 200
		cache.CopyPrefix(srcSeq, dstSeq, 128)
	}
}

// BenchmarkPagedCache_BlockSizeVariation benchmarks performance with different block sizes.
func BenchmarkPagedCache_BlockSizeVariation(b *testing.B) {
	blockSizes := []int{8, 16, 32, 64, 128}

	for _, bs := range blockSizes {
		b.Run(bs_name(bs), func(b *testing.B) {
			cache := NewPagedCacheWithConfig(nil, bs, 10000)
			cache.Init(&mockBackend{}, ml.DTypeF32, 100, 2048, 100)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				seqID := i % 100
				cache.allocateBlocksForSequence(seqID, 256)
			}
		})
	}
}

// BenchmarkPagedCache_MemoryFragmentation simulates fragmented allocation patterns.
func BenchmarkPagedCache_MemoryFragmentation(b *testing.B) {
	cache := NewPagedCacheWithConfig(nil, 16, 10000)
	cache.Init(&mockBackend{}, ml.DTypeF32, 100, 256, 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Allocate
		seqID := i % 100
		cache.allocateBlocksForSequence(seqID, 32)

		// Randomly free some sequences to create fragmentation
		if i%10 == 0 {
			freeSeq := (i + 50) % 100
			cache.Remove(freeSeq, 0, math.MaxInt32)
		}
	}
}

// BenchmarkPagedCache_VaryingSequenceLengths benchmarks with varying sequence lengths.
func BenchmarkPagedCache_VaryingSequenceLengths(b *testing.B) {
	cache := NewPagedCacheWithConfig(nil, 16, 10000)
	cache.Init(&mockBackend{}, ml.DTypeF32, 100, 2048, 100)

	lengths := []int{16, 32, 64, 128, 256, 512, 1024, 2048}

	for _, length := range lengths {
		b.Run(length_name(length), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				seqID := i % 100
				cache.allocateBlocksForSequence(seqID, length)
			}
		})
	}
}

// BenchmarkPagedCache_ConcurrentAccess benchmarks concurrent access patterns.
func BenchmarkPagedCache_ConcurrentAccess(b *testing.B) {
	cache := NewPagedCacheWithConfig(nil, 16, 10000)
	cache.Init(&mockBackend{}, ml.DTypeF32, 100, 256, 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Simulate concurrent operations
		seqID := i % 100
		cache.allocateBlocksForSequence(seqID, 32)
		_ = cache.CanResume(seqID, 16)
		_ = cache.GetStats()
	}
}

// Helper functions for naming
func bs_name(bs int) string {
	switch bs {
	case 8:
		return "block_size_8"
	case 16:
		return "block_size_16"
	case 32:
		return "block_size_32"
	case 64:
		return "block_size_64"
	case 128:
		return "block_size_128"
	default:
		return "unknown"
	}
}

func length_name(l int) string {
	switch l {
	case 16:
		return "16_tokens"
	case 32:
		return "32_tokens"
	case 64:
		return "64_tokens"
	case 128:
		return "128_tokens"
	case 256:
		return "256_tokens"
	case 512:
		return "512_tokens"
	case 1024:
		return "1k_tokens"
	case 2048:
		return "2k_tokens"
	default:
		return "unknown"
	}
}
