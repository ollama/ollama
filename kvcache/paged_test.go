package kvcache

import (
	"math"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

func TestNewPagedCache(t *testing.T) {
	cache := NewPagedCache(nil)
	if cache == nil {
		t.Fatal("NewPagedCache returned nil")
	}

	if cache.blockSize != DefaultBlockSize {
		t.Errorf("expected block size %d, got %d", DefaultBlockSize, cache.blockSize)
	}

	if cache.maxNumBlocks != DefaultMaxNumBlocks {
		t.Errorf("expected max num blocks %d, got %d", DefaultMaxNumBlocks, cache.maxNumBlocks)
	}
}

func TestNewPagedCacheWithConfig(t *testing.T) {
	blockSize := 32
	maxNumBlocks := 500

	cache := NewPagedCacheWithConfig(nil, blockSize, maxNumBlocks)

	if cache.blockSize != blockSize {
		t.Errorf("expected block size %d, got %d", blockSize, cache.blockSize)
	}

	if cache.maxNumBlocks != maxNumBlocks {
		t.Errorf("expected max num blocks %d, got %d", maxNumBlocks, cache.maxNumBlocks)
	}
}

func TestPagedCache_Init(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	// Verify initialization
	if cache.DType != ml.DTypeF16 {
		t.Errorf("expected dtype %v, got %v", ml.DTypeF16, cache.DType)
	}

	if cache.maxSequences != 10 {
		t.Errorf("expected max sequences 10, got %d", cache.maxSequences)
	}

	if cache.capacity != 1000 {
		t.Errorf("expected capacity 1000, got %d", cache.capacity)
	}

	if cache.numBlocks == 0 {
		t.Error("numBlocks should be > 0 after Init")
	}

	// All blocks should be free initially
	expectedFree := cache.numBlocks
	if len(cache.freeBlocks) != expectedFree {
		t.Errorf("expected %d free blocks, got %d", expectedFree, len(cache.freeBlocks))
	}
}

func TestPagedCache_AllocateBlock(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	initialFree := len(cache.freeBlocks)

	// Allocate a block
	blockID, err := cache.allocateBlock()
	if err != nil {
		t.Fatalf("allocateBlock failed: %v", err)
	}

	if blockID < 0 || blockID >= cache.numBlocks {
		t.Errorf("allocated block ID %d out of range [0, %d)", blockID, cache.numBlocks)
	}

	// Free blocks should decrease by 1
	if len(cache.freeBlocks) != initialFree-1 {
		t.Errorf("expected %d free blocks, got %d", initialFree-1, len(cache.freeBlocks))
	}

	// Allocated blocks should increase
	if cache.allocatedBlocks != 1 {
		t.Errorf("expected 1 allocated block, got %d", cache.allocatedBlocks)
	}
}

func TestPagedCache_AllocateBlockExhausted(t *testing.T) {
	cache := NewPagedCacheWithConfig(nil, 16, 2) // Only 2 blocks
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	// Allocate all blocks
	_, err1 := cache.allocateBlock()
	if err1 != nil {
		t.Fatalf("first allocateBlock failed: %v", err1)
	}

	_, err2 := cache.allocateBlock()
	if err2 != nil {
		t.Fatalf("second allocateBlock failed: %v", err2)
	}

	// Third allocation should fail
	_, err3 := cache.allocateBlock()
	if err3 == nil {
		t.Error("expected error when allocating beyond capacity")
	}
}

func TestPagedCache_FreeBlock(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	// Allocate a block
	blockID, _ := cache.allocateBlock()

	initialFree := len(cache.freeBlocks)
	initialAllocated := cache.allocatedBlocks

	// Free the block
	cache.freeBlock(blockID)

	// Free blocks should increase
	if len(cache.freeBlocks) != initialFree+1 {
		t.Errorf("expected %d free blocks, got %d", initialFree+1, len(cache.freeBlocks))
	}

	// Allocated blocks should decrease
	if cache.allocatedBlocks != initialAllocated-1 {
		t.Errorf("expected %d allocated blocks, got %d", initialAllocated-1, cache.allocatedBlocks)
	}

	// Block mapping should be cleared
	if cache.blockMapping[blockID].seqID != -1 {
		t.Error("block mapping should be cleared after free")
	}
}

func TestPagedCache_GetNumBlocksForTokens(t *testing.T) {
	cache := NewPagedCacheWithConfig(nil, 16, 1000)

	tests := []struct {
		tokens       int
		expectedBlocks int
	}{
		{1, 1},
		{16, 1},
		{17, 2},
		{32, 2},
		{33, 3},
		{100, 7}, // ceil(100/16) = 7
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			blocks := cache.getNumBlocksForTokens(tt.tokens)
			if blocks != tt.expectedBlocks {
				t.Errorf("tokens=%d: expected %d blocks, got %d",
					tt.tokens, tt.expectedBlocks, blocks)
			}
		})
	}
}

func TestPagedCache_AllocateBlocksForSequence(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	seqID := 1
	numTokens := 50 // Should allocate ceil(50/16) = 4 blocks

	blocks, err := cache.allocateBlocksForSequence(seqID, numTokens)
	if err != nil {
		t.Fatalf("allocateBlocksForSequence failed: %v", err)
	}

	expectedBlocks := (numTokens + cache.blockSize - 1) / cache.blockSize
	if len(blocks) != expectedBlocks {
		t.Errorf("expected %d blocks, got %d", expectedBlocks, len(blocks))
	}

	// Block table should be updated
	storedBlocks, ok := cache.blockTables[seqID]
	if !ok {
		t.Fatal("block table entry not created for sequence")
	}

	if len(storedBlocks) != len(blocks) {
		t.Errorf("block table has %d blocks, expected %d", len(storedBlocks), len(blocks))
	}

	// Each block should have the correct mapping
	for i, blockID := range blocks {
		entry := cache.blockMapping[blockID]
		if entry.seqID != seqID {
			t.Errorf("block %d: expected seqID %d, got %d", i, seqID, entry.seqID)
		}
		if entry.logicalBlock != i {
			t.Errorf("block %d: expected logical block %d, got %d", i, i, entry.logicalBlock)
		}
	}
}

func TestPagedCache_AllocateBlocksForSequenceExhausted(t *testing.T) {
	cache := NewPagedCacheWithConfig(nil, 16, 3) // Only 3 blocks
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	seqID := 1
	numTokens := 100 // Needs 7 blocks but we only have 3

	_, err := cache.allocateBlocksForSequence(seqID, numTokens)
	if err == nil {
		t.Error("expected error when allocating beyond capacity")
	}

	// All blocks should be freed after failed allocation
	if len(cache.freeBlocks) != cache.numBlocks {
		t.Errorf("expected all blocks to be free, got %d/%d free",
			len(cache.freeBlocks), cache.numBlocks)
	}
}

func TestPagedCache_CanResume(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	seqID := 1
	numTokens := 50
	cache.allocateBlocksForSequence(seqID, numTokens)

	tests := []struct {
		pos      int32
		expected bool
	}{
		{0, true},
		{15, true},   // First block
		{16, true},   // Second block
		{31, true},   // End of second block
		{32, true},   // Third block
		{48, true},   // Last position in 4th block
		{49, true},   // Last valid position
		{50, true},   // Within 4th block (16*3=48 to 16*4-1=63)
		{100, false}, // Way beyond
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			result := cache.CanResume(seqID, tt.pos)
			if result != tt.expected {
				t.Errorf("pos=%d: expected %v, got %v", tt.pos, tt.expected, result)
			}
		})
	}
}

func TestPagedCache_CanResume_NoSequence(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	// Non-existent sequence
	result := cache.CanResume(999, 10)
	if result {
		t.Error("expected false for non-existent sequence")
	}
}

func TestPagedCache_Remove(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	seqID := 1
	numTokens := 100
	cache.allocateBlocksForSequence(seqID, numTokens)

	initialAllocated := cache.allocatedBlocks

	// Remove tokens in range [32, 64) (blocks 2 and 3)
	err := cache.Remove(seqID, 32, 64)
	if err != nil {
		t.Fatalf("Remove failed: %v", err)
	}

	// Two blocks should be freed
	if cache.allocatedBlocks != initialAllocated-2 {
		t.Errorf("expected %d allocated blocks, got %d",
			initialAllocated-2, cache.allocatedBlocks)
	}

	// Block table should be updated
	blocks, ok := cache.blockTables[seqID]
	if !ok {
		t.Fatal("block table entry missing after Remove")
	}

	expectedBlocks := 5 // Original 7 blocks, minus 2 removed (blocks 2 and 3)
	if len(blocks) != expectedBlocks {
		t.Errorf("expected %d blocks after Remove, got %d", expectedBlocks, len(blocks))
	}
}

func TestPagedCache_RemoveAll(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	seqID := 1
	numTokens := 100
	cache.allocateBlocksForSequence(seqID, numTokens)

	// Remove all tokens from position 16 onwards
	err := cache.Remove(seqID, 16, math.MaxInt32)
	if err != nil {
		t.Fatalf("Remove failed: %v", err)
	}

	// Should have 1 block remaining (tokens 0-15)
	blocks, ok := cache.blockTables[seqID]
	if !ok {
		t.Fatal("block table entry missing after Remove")
	}

	if len(blocks) != 1 {
		t.Errorf("expected 1 block after Remove all, got %d", len(blocks))
	}
}

func TestPagedCache_Remove_NoSequence(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	// Removing from non-existent sequence should not error
	err := cache.Remove(999, 0, 10)
	if err != nil {
		t.Errorf("Remove on non-existent sequence should not error, got: %v", err)
	}
}

func TestPagedCache_GetStats(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	stats := cache.GetStats()

	if stats.NumBlocks != cache.numBlocks {
		t.Errorf("expected NumBlocks %d, got %d", cache.numBlocks, stats.NumBlocks)
	}

	if stats.AllocatedBlocks != cache.allocatedBlocks {
		t.Errorf("expected AllocatedBlocks %d, got %d", cache.allocatedBlocks, stats.AllocatedBlocks)
	}

	if stats.FreeBlocks != len(cache.freeBlocks) {
		t.Errorf("expected FreeBlocks %d, got %d", len(cache.freeBlocks), stats.FreeBlocks)
	}

	if stats.BlockSize != cache.blockSize {
		t.Errorf("expected BlockSize %d, got %d", cache.blockSize, stats.BlockSize)
	}

	if stats.ActiveSequences != len(cache.blockTables) {
		t.Errorf("expected ActiveSequences %d, got %d", len(cache.blockTables), stats.ActiveSequences)
	}

	// Utilization should be 0 initially
	if stats.Utilization != 0 {
		t.Errorf("expected 0 utilization, got %f", stats.Utilization)
	}

	// Allocate some blocks
	cache.allocateBlocksForSequence(1, 50)
	cache.allocateBlocksForSequence(2, 30)

	stats = cache.GetStats()

	expectedUtil := float64(cache.allocatedBlocks) / float64(cache.numBlocks)
	if stats.Utilization != expectedUtil {
		t.Errorf("expected utilization %f, got %f", expectedUtil, stats.Utilization)
	}

	if stats.ActiveSequences != 2 {
		t.Errorf("expected 2 active sequences, got %d", stats.ActiveSequences)
	}
}

func TestPagedCache_StartForward(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	context := backend.NewContext()
	defer context.Close()

	batch := input.Batch{
		Sequences:    []int{1, 1, 2},
		Positions: []int32{0, 1, 0},
	}

	err := cache.StartForward(context, batch, false)
	if err != nil {
		t.Fatalf("StartForward failed: %v", err)
	}

	if cache.curBatchSize != len(batch.Sequences) {
		t.Errorf("expected batch size %d, got %d", len(batch.Sequences), cache.curBatchSize)
	}

	// Blocks should be allocated for both sequences
	if len(cache.blockTables) != 2 {
		t.Errorf("expected 2 sequences in block table, got %d", len(cache.blockTables))
	}
}

func TestPagedCache_StartForward_InsufficientCapacity(t *testing.T) {
	cache := NewPagedCacheWithConfig(nil, 16, 2) // Only 2 blocks = 32 tokens
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 100, 32)

	batch := input.Batch{
		Sequences:    []int{1, 2},
		Positions: []int32{50, 50}, // Need 4 blocks but only have 2
	}

	err := cache.StartForward(nil, batch, false)
	if err == nil {
		t.Error("expected error for insufficient capacity")
	}
}

func TestPagedCache_CopyPrefix(t *testing.T) {
	cache := NewPagedCache(nil)
	backend := &testBackend{}

	cache.Init(backend, ml.DTypeF16, 10, 1000, 32)

	srcSeq := 1
	dstSeq := 2
	length := int32(50)

	// Allocate source sequence
	cache.allocateBlocksForSequence(srcSeq, int(length))

	// Copy prefix
	cache.CopyPrefix(srcSeq, dstSeq, length)

	// Destination should have blocks allocated
	dstBlocks, ok := cache.blockTables[dstSeq]
	if !ok {
		t.Fatal("destination sequence not in block table after CopyPrefix")
	}

	srcBlocks := cache.blockTables[srcSeq]
	expectedBlocks := len(srcBlocks)
	if len(dstBlocks) != expectedBlocks {
		t.Errorf("expected %d blocks for destination, got %d", expectedBlocks, len(dstBlocks))
	}
}
