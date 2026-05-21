//go:build benchmark
// +build benchmark

package kvcache

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/ml"
)

// BenchmarkPerformanceCompare compares original vs accelerated implementation.
// Run with: go test -tags=benchmark -bench=. -run=^$
func BenchmarkPerformanceCompare(b *testing.B) {
	capacities := []int{1000, 10000, 100000}
	batchSizes := []int{1, 10, 100}

	fmt.Println("\n=== Performance Comparison ===")
	fmt.Println("Capacity | Batch | Original (ns) | Accelerated (ns) | Speedup")
	fmt.Println("---------|-------|---------------|------------------|--------")

	for _, capacity := range capacities {
		for _, batchSize := range batchSizes {
			// Benchmark original implementation
			tOriginal := testing.Benchmark(func(b *testing.B) {
				cache := NewCausalCache(nil)
				cache.Init(&testBackend{}, ml.DTypeF16, 1, capacity, batchSize)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					cache.curBatchSize = batchSize
					_, _ = cache.findLocs()
				}
			})

			// Benchmark accelerated implementation
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
