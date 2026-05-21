package kvcache

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/ml"
)

// BenchmarkFindLocsOriginal benchmarks the original implementation
func BenchmarkFindLocsOriginal(b *testing.B) {
	benchmarks := []struct {
		capacity  int
		batchSize int
	}{
		{1000, 1}, {1000, 10}, {1000, 100},
		{10000, 1}, {10000, 10}, {10000, 100},
		{100000, 1}, {100000, 10}, {100000, 100},
	}

	for _, bm := range benchmarks {
		b.Run(fmt.Sprintf("c%d_b%d", bm.capacity, bm.batchSize), func(b *testing.B) {
			cache := NewCausalCache(nil)
			cache.Init(&testBackend{}, ml.DTypeF16, 1, bm.capacity, bm.batchSize)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cache.curBatchSize = bm.batchSize
				_, _ = cache.findLocs()
			}
		})
	}
}

// BenchmarkFindLocsAccelerated benchmarks the accelerated implementation
func BenchmarkFindLocsAccelerated(b *testing.B) {
	benchmarks := []struct {
		capacity  int
		batchSize int
	}{
		{1000, 1}, {1000, 10}, {1000, 100},
		{10000, 1}, {10000, 10}, {10000, 100},
		{100000, 1}, {100000, 10}, {100000, 100},
	}

	for _, bm := range benchmarks {
		b.Run(fmt.Sprintf("c%d_b%d", bm.capacity, bm.batchSize), func(b *testing.B) {
			cache := NewAcceleratedCache(nil)
			cache.Init(&testBackend{}, ml.DTypeF16, 1, bm.capacity, bm.batchSize)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cache.curBatchSize = bm.batchSize
				_, _ = cache.findLocsAccelerated()
			}
		})
	}
}
