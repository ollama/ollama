package kvcache

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/ml"
)

// TestPerformanceCompare compares original vs accelerated implementation
func TestPerformanceCompare(t *testing.T) {
	capacities := []int{1000, 10000, 100000}
	batchSizes := []int{1, 10, 100}

	fmt.Println("\n=== Performance Comparison ===")
	fmt.Println("Capacity | Batch | Original (ns) | Accelerated (ns) | Speedup")
	fmt.Println("---------|-------|---------------|------------------|--------")

	for _, capacity := range capacities {
		for _, batchSize := range batchSizes {
			// Test original implementation
			originalCache := NewCausalCache(nil)
			originalCache.Init(&testBackend{}, ml.DTypeF16, 1, capacity, batchSize)

			// Warm up
			for i := 0; i < 100; i++ {
				originalCache.curBatchSize = batchSize
				_, _ = originalCache.findLocs()
			}

			// Measure original
			tOriginal := testing.Benchmark(func(b *testing.B) {
				cache := NewCausalCache(nil)
				cache.Init(&testBackend{}, ml.DTypeF16, 1, capacity, batchSize)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					cache.curBatchSize = batchSize
					_, _ = cache.findLocs()
				}
			})

			// Test accelerated implementation
			accelCache := NewAcceleratedCache(nil)
			accelCache.Init(&testBackend{}, ml.DTypeF16, 1, capacity, batchSize)

			// Warm up
			for i := 0; i < 100; i++ {
				accelCache.curBatchSize = batchSize
				_, _ = accelCache.findLocsAccelerated()
			}

			// Measure accelerated
			tAccel := testing.Benchmark(func(b *testing.B) {
				cache := NewAcceleratedCache(nil)
				cache.Init(&testBackend{}, ml.DTypeF16, 1, capacity, batchSize)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					cache.curBatchSize = batchSize
					_, _ = cache.findLocsAccelerated()
				}
			})

			speedup := float64(tOriginal.NsPerOp()) / float64(tAccel.NsPerOp())
			fmt.Printf("%8d | %5d | %13d | %15d | %7.1fx\n",
				capacity, batchSize, tOriginal.NsPerOp(), tAccel.NsPerOp(), speedup)
		}
	}
}
