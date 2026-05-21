package kvcache

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// TestAcceleratedFindLocs tests the O(1) free block pool
func TestAcceleratedFindLocs(t *testing.T) {
	tests := []struct {
		name          string
		capacity      int
		batchSize     int
		numAllocs     int
		wantErr       bool
	}{
		{
			name:      "SingleAllocation",
			capacity:  100,
			batchSize: 10,
			numAllocs: 1,
			wantErr:   false,
		},
		{
			name:      "MultipleAllocations",
			capacity:  100,
			batchSize: 10,
			numAllocs: 3,
			wantErr:   false,
		},
		{
			name:      "CacheFull",
			capacity:  20,
			batchSize: 10,
			numAllocs: 3, // Will try to allocate 30 total but only have 20
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache := NewAcceleratedCache(nil)
			defer cache.Close()

			cache.Init(&testBackend{}, ml.DTypeF16, 1, tt.capacity, tt.batchSize)

			var err error
			for i := 0; i < tt.numAllocs; i++ {
				cache.curBatchSize = tt.batchSize
				var locs []int32
				locs, err = cache.findLocsAccelerated()

				if err != nil {
					if tt.wantErr && i == tt.numAllocs-1 {
						// Expected error on last allocation
						return
					}
					t.Errorf("unexpected error for allocation %d: %v", i, err)
					return
				}

				// Mark allocated cells as occupied
				for _, loc := range locs {
					cache.clearBit(int(loc))
				}
			}

			if tt.wantErr {
				t.Errorf("expected error but all allocations succeeded")
			}
		})
	}
}

// TestAcceleratedReuse tests that freed cells can be reused
func TestAcceleratedReuse(t *testing.T) {
	cache := NewAcceleratedCache(nil)
	defer cache.Close()

	cache.Init(&testBackend{}, ml.DTypeF16, 1, 100, 10)

	// Allocate first batch
	cache.curBatchSize = 10
	locs1, err := cache.findLocsAccelerated()
	if err != nil {
		t.Fatalf("first allocation failed: %v", err)
	}

	// Mark them as occupied
	for _, loc := range locs1 {
		cache.clearBit(int(loc))
	}

	// Allocate second batch
	cache.curBatchSize = 10
	locs2, err := cache.findLocsAccelerated()
	if err != nil {
		t.Fatalf("second allocation failed: %v", err)
	}

	// Verify they don't overlap
	for _, l1 := range locs1 {
		for _, l2 := range locs2 {
			if l1 == l2 {
				t.Errorf("locations overlap: %d", l1)
			}
		}
	}

	// Free first batch
	for _, loc := range locs1 {
		cache.setBit(int(loc))
	}

	// Allocate third batch - should reuse first batch's locations
	cache.curBatchSize = 10
	locs3, err := cache.findLocsAccelerated()
	if err != nil {
		t.Fatalf("third allocation failed: %v", err)
	}

	// Verify third batch reuses first batch's locations
	reused := 0
	for _, l3 := range locs3 {
		for _, l1 := range locs1 {
			if l3 == l1 {
				reused++
				break
			}
		}
	}

	if reused != len(locs1) {
		t.Errorf("expected to reuse all %d locations from first batch, only reused %d", len(locs1), reused)
	}
}

// TestAcceleratedBitmapOperations tests bitmap operations
func TestAcceleratedBitmapOperations(t *testing.T) {
	cache := NewAcceleratedCache(nil)
	defer cache.Close()

	cache.Init(&testBackend{}, ml.DTypeF16, 1, 1000, 10)

	if cache.freeCount != 1000 {
		t.Errorf("expected freeCount to be 1000, got %d", cache.freeCount)
	}

	locations := []int{10, 20, 30, 40, 50}
	for _, loc := range locations {
		cache.clearBit(loc)
	}

	expectedFree := 1000 - len(locations)
	if cache.freeCount != expectedFree {
		t.Errorf("expected freeCount to be %d, got %d", expectedFree, cache.freeCount)
	}

	for _, loc := range locations {
		if cache.isSet(loc) {
			t.Errorf("location %d should be occupied but is marked free", loc)
		}
	}

	for _, loc := range locations {
		cache.setBit(loc)
	}

	if cache.freeCount != 1000 {
		t.Errorf("expected freeCount to be 1000 after freeing, got %d", cache.freeCount)
	}

	for _, loc := range locations {
		if !cache.isSet(loc) {
			t.Errorf("location %d should be free but is marked occupied", loc)
		}
	}
}

// TestAcceleratedStartForward tests the StartForward method
func TestAcceleratedStartForward(t *testing.T) {
	runPermutedVariants(t, func(t *testing.T, backend *testBackend) {
		cache := NewAcceleratedCache(nil)
		defer cache.Close()

		cache.Init(backend, ml.DTypeF16, 1, 16, 4)

		batch := input.Batch{
			Sequences: []int{0, 0, 0, 0},
			Positions: []int32{0, 1, 2, 3},
		}

		ctx := &testContext{}
		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			t.Fatalf("StartForward failed: %v", err)
		}

		if cache.freeCount != 12 {
			t.Errorf("expected 12 free cells after allocation, got %d", cache.freeCount)
		}

		for i := 0; i < 4; i++ {
			if cache.isSet(i) {
				t.Errorf("cell %d should be occupied", i)
			}
		}

		batch2 := input.Batch{
			Sequences: []int{0},
			Positions: []int32{4},
		}

		err = cache.StartForward(ctx, batch2, false)
		if err != nil {
			t.Fatalf("second StartForward failed: %v", err)
		}

		if cache.freeCount != 11 {
			t.Errorf("expected 11 free cells after second allocation, got %d", cache.freeCount)
		}
	})
}

// BenchmarkFindLocs compares original vs accelerated implementation
func BenchmarkFindLocs(b *testing.B) {
	capacities := []int{1000, 10000, 100000}
	batchSizes := []int{1, 10, 100}

	for _, capacity := range capacities {
		for _, batchSize := range batchSizes {
			b.Run(fmt.Sprintf("Accelerated_c%d_b%d", capacity, batchSize), func(b *testing.B) {
				cache := NewAcceleratedCache(nil)
				cache.Init(&testBackend{}, ml.DTypeF16, 1, capacity, batchSize)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					cache.curBatchSize = batchSize
					_, _ = cache.findLocsAccelerated()
				}
			})
		}
	}
}

// BenchmarkBitmapOperations benchmarks bitmap operations
func BenchmarkBitmapOperations(b *testing.B) {
	cache := NewAcceleratedCache(nil)
	cache.Init(&testBackend{}, ml.DTypeF16, 1, 100000, 10)

	for i := 0; i < 50000; i += 2 {
		cache.clearBit(i)
	}

	b.Run("IsSet", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			cache.isSet(i % 100000)
		}
	})

	b.Run("SetBit", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			loc := (i * 2) % 100000
			if !cache.isSet(loc) {
				cache.setBit(loc)
			}
		}
	})

	b.Run("ClearBit", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			loc := (i * 2 + 1) % 100000
			if cache.isSet(loc) {
				cache.clearBit(loc)
			}
		}
	})
}
