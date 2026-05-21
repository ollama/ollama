package kvcache

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/ml"
)

// BenchmarkRealisticScenario tests with realistic allocation patterns
// Simulates: alloc -> use -> free cycle
func BenchmarkRealisticScenario(b *testing.B) {
	scenarios := []struct {
		name      string
		capacity  int
		batchSize int
		fillRatio float64
	}{
		{"Empty_c1000_b10", 1000, 10, 0.0},
		{"HalfFull_c1000_b10", 1000, 10, 0.5},
		{"MostlyFull_c1000_b10", 1000, 10, 0.9},
		{"HalfFull_c10000_b10", 10000, 10, 0.5},
		{"MostlyFull_c10000_b10", 10000, 10, 0.9},
		{"HalfFull_c100000_b10", 100000, 10, 0.5},
		{"MostlyFull_c100000_b10", 100000, 10, 0.9},
	}

	for _, sc := range scenarios {
		b.Run("Original_"+sc.name, func(b *testing.B) {
			cache := NewCausalCache(nil)
			cache.Init(&testBackend{}, ml.DTypeF16, 1, sc.capacity, sc.batchSize)

			// Pre-fill: occupy first N slots with data
			fillCount := int(float64(sc.capacity) * sc.fillRatio)
			for i := 0; i < fillCount; i++ {
				cache.cells[i] = cacheCell{pos: int32(i), sequences: []int{0}}
			}

			// Track allocated slots for cleanup
			var lastAlloc []int32

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cache.curBatchSize = sc.batchSize

				// Free previous allocation
				for _, loc := range lastAlloc {
					if int(loc) < len(cache.cells) {
						cache.cells[loc] = cacheCell{}
					}
				}

				// Allocate new slots
				locs, _ := cache.findLocs()
				lastAlloc = locs

				// Mark as occupied
				for _, loc := range locs {
					cache.cells[loc] = cacheCell{pos: 999, sequences: []int{1}}
				}
			}
		})

		b.Run("Accelerated_"+sc.name, func(b *testing.B) {
			cache := NewAcceleratedCache(nil)
			cache.Init(&testBackend{}, ml.DTypeF16, 1, sc.capacity, sc.batchSize)

			// Pre-fill: occupy first N slots
			fillCount := int(float64(sc.capacity) * sc.fillRatio)
			for i := 0; i < fillCount; i++ {
				cache.clearBit(i)
				cache.cells[i] = cacheCell{pos: int32(i), sequences: []int{0}}
			}

			// Track allocated slots for cleanup
			var lastAlloc []int32

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cache.curBatchSize = sc.batchSize

				// Free previous allocation
				for _, loc := range lastAlloc {
					if int(loc) < len(cache.cells) {
						cache.setBit(int(loc))
						cache.cells[loc] = cacheCell{}
					}
				}

				// Allocate new slots
				locs, _ := cache.findLocsAccelerated()
				lastAlloc = locs

				// Mark as occupied
				for _, loc := range locs {
					cache.clearBit(int(loc))
					cache.cells[loc] = cacheCell{pos: 999, sequences: []int{1}}
				}
			}
		})
	}
}

// BenchmarkWorstCase tests the worst case: finding slots when cache is fragmented
func BenchmarkWorstCase(b *testing.B) {
	capacities := []int{1000, 10000, 100000}

	for _, capacity := range capacities {
		b.Run("Original_"+fmt.Sprint(capacity), func(b *testing.B) {
			cache := NewCausalCache(nil)
			cache.Init(&testBackend{}, ml.DTypeF16, 1, capacity, 10)

			// Create fragmented pattern: occupy every other slot
			for i := 0; i < capacity; i += 2 {
				cache.cells[i] = cacheCell{pos: int32(i), sequences: []int{0}}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cache.curBatchSize = 10
				_, _ = cache.findLocs()
			}
		})

		b.Run("Accelerated_"+fmt.Sprint(capacity), func(b *testing.B) {
			cache := NewAcceleratedCache(nil)
			cache.Init(&testBackend{}, ml.DTypeF16, 1, capacity, 10)

			// Create same fragmented pattern
			for i := 0; i < capacity; i += 2 {
				cache.clearBit(i)
				cache.cells[i] = cacheCell{pos: int32(i), sequences: []int{0}}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cache.curBatchSize = 10
				_, _ = cache.findLocsAccelerated()
			}
		})
	}
}
