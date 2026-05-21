package kvcache

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// BenchmarkPagedCacheAllocation benchmarks block allocation performance.
func BenchmarkPagedCacheAllocation(b *testing.B) {
	blockSizes := []int{8, 16, 32}

	for _, blockSize := range blockSizes {
		b.Run(fmt.Sprintf("BlockSize_%d", blockSize), func(b *testing.B) {
			cache := NewPagedCacheWithConfig(nil, blockSize, 10000)
			backend := &testBackend{}
			cache.Init(backend, ml.DTypeF16, 100, 10000, 32)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				seqID := i % 100
				cache.allocateBlocksForSequence(seqID, 100)

				// Clean up periodically to prevent exhaustion
				if i%100 == 99 {
					cache.blockTables = make(map[int][]int)
					cache.allocatedBlocks = 0
					cache.freeBlocks = make([]int, cache.numBlocks)
					for j := 0; j < cache.numBlocks; j++ {
						cache.freeBlocks[j] = j
					}
				}
			}
		})
	}
}

// BenchmarkPagedCacheStartForward benchmarks the StartForward operation.
func BenchmarkPagedCacheStartForward(b *testing.B) {
	batchSizes := []int{1, 8, 16, 32}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("BatchSize_%d", batchSize), func(b *testing.B) {
			cache := NewPagedCache(nil)
			backend := &testBackend{}
			cache.Init(backend, ml.DTypeF16, 100, 10000, batchSize)

			// Pre-allocate some sequences
			for seqID := 0; seqID < 10; seqID++ {
				cache.allocateBlocksForSequence(seqID, 100)
			}

			batch := input.Batch{
				Sequences:    make([]int, batchSize),
				Positions: make([]int32, batchSize),
			}
			for i := 0; i < batchSize; i++ {
				batch.Sequences[i] = i % 10
				batch.Positions[i] = int32(i / 10)
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				context := backend.NewContext(); defer context.Close(); cache.StartForward(context, batch, false)
			}
		})
	}
}

// BenchmarkPagedCacheVsCausal compares paged vs causal cache performance.
func BenchmarkPagedCacheVsCausal(b *testing.B) {
	scenarios := []struct {
		name        string
		maxSeqs     int
		capacity    int
		batchSize   int
		numTokens   int
	}{
		{"Small", 10, 1000, 8, 50},
		{"Medium", 50, 5000, 16, 200},
		{"Large", 100, 10000, 32, 500},
	}

	for _, scenario := range scenarios {
		b.Run(scenario.name, func(b *testing.B) {
			b.Run("Paged", func(b *testing.B) {
				b.Run("StartForward", func(b *testing.B) {
					cache := NewPagedCache(nil)
					backend := &testBackend{}
					cache.Init(backend, ml.DTypeF16, scenario.maxSeqs, scenario.capacity, scenario.batchSize)

					// Pre-allocate sequences
					for seqID := 0; seqID < scenario.maxSeqs/2; seqID++ {
						cache.allocateBlocksForSequence(seqID, scenario.numTokens)
					}

					batch := input.Batch{
						Sequences:    make([]int, scenario.batchSize),
						Positions: make([]int32, scenario.batchSize),
					}
					for i := 0; i < scenario.batchSize; i++ {
						batch.Sequences[i] = i % (scenario.maxSeqs / 2)
						batch.Positions[i] = int32(i / (scenario.maxSeqs / 2))
					}

					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						context := backend.NewContext(); defer context.Close(); cache.StartForward(context, batch, false)
					}
				})
			})

			b.Run("Causal", func(b *testing.B) {
				b.Run("StartForward", func(b *testing.B) {
					cache := NewCausalCache(nil)
					backend := &testBackend{}
					cache.Init(backend, ml.DTypeF16, scenario.maxSeqs, scenario.capacity, scenario.batchSize)

					batch := input.Batch{
						Sequences:    make([]int, scenario.batchSize),
						Positions: make([]int32, scenario.batchSize),
					}
					for i := 0; i < scenario.batchSize; i++ {
						batch.Sequences[i] = i % (scenario.maxSeqs / 2)
						batch.Positions[i] = int32(i / (scenario.maxSeqs / 2))
					}

					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						context := backend.NewContext(); defer context.Close(); cache.StartForward(context, batch, false)
					}
				})
			})
		})
	}
}

// BenchmarkPagedCacheBlockReuse benchmarks block reuse performance.
func BenchmarkPagedCacheBlockReuse(b *testing.B) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}
	cache.Init(backend, ml.DTypeF16, 100, 10000, 32)

	// Allocate initial blocks
	for seqID := 0; seqID < 10; seqID++ {
		cache.allocateBlocksForSequence(seqID, 100)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Remove and reallocate blocks
		seqID := i % 10
		cache.Remove(seqID, 0, 50)
		cache.allocateBlocksForSequence(seqID, 100)
	}
}

// BenchmarkPagedCacheMemoryOverhead benchmarks the memory overhead of paged cache.
func BenchmarkPagedCacheMemoryOverhead(b *testing.B) {
	b.Run("Paged", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			cache := NewPagedCache(nil)
			backend := &testBackend{}
			cache.Init(backend, ml.DTypeF16, 100, 10000, 32)
			_ = cache
		}
	})

	b.Run("Causal", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			cache := NewCausalCache(nil)
			backend := &testBackend{}
			cache.Init(backend, ml.DTypeF16, 100, 10000, 32)
			_ = cache
		}
	})
}

// BenchmarkPagedCacheFragmentation simulates memory fragmentation scenarios.
func BenchmarkPagedCacheFragmentation(b *testing.B) {
	b.Run("Paged_NoFragmentation", func(b *testing.B) {
		cache := NewPagedCache(nil)
		backend := &testBackend{}
		cache.Init(backend, ml.DTypeF16, 100, 10000, 32)

		// Allocate continuous sequences
		for seqID := 0; seqID < 10; seqID++ {
			cache.allocateBlocksForSequence(seqID, 100)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			seqID := i % 10
			cache.allocateBlocksForSequence(seqID+100, 50)
		}
	})

	b.Run("Paged_WithFragmentation", func(b *testing.B) {
		cache := NewPagedCache(nil)
		backend := &testBackend{}
		cache.Init(backend, ml.DTypeF16, 100, 10000, 32)

		// Create fragmented allocation pattern
		for seqID := 0; seqID < 50; seqID++ {
			cache.allocateBlocksForSequence(seqID, 20)
		}
		// Free every other sequence
		for seqID := 0; seqID < 50; seqID += 2 {
			cache.Remove(seqID, 0, 20)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			seqID := (i % 25) * 2 // Allocate in gaps
			cache.allocateBlocksForSequence(seqID+100, 20)
		}
	})
}
