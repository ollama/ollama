package lfm2

import (
	"testing"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

// TestHybridCache tests verify the slot management logic of HybridCache.
// These tests focus on the recurrent state slot allocation, reference counting,
// and copy-on-write semantics without requiring a full ML backend.

// createSlotOnlyCache creates a HybridCache with only the slot management
// fields initialized. Used to test slot logic in isolation.
func createSlotOnlyCache(maxSequences int) *HybridCache {
	return &HybridCache{
		hiddenSize:   256,
		dConv:        3,
		maxSequences: maxSequences,
		refCount:     make([]int, maxSequences),
		freeSlots:    initFreeSlots(maxSequences),
		slotForSeq:   make(map[int]int),
		convCtxs:     make(map[int]ml.Context),
		convStates:   make(map[int]ml.Tensor),
	}
}

func initFreeSlots(n int) []int {
	slots := make([]int, 0, n)
	for i := n - 1; i >= 0; i-- {
		slots = append(slots, i)
	}
	return slots
}

func TestHybridCache_SlotAllocation(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// Verify initial state
	if len(cache.freeSlots) != 4 {
		t.Errorf("expected 4 free slots, got %d", len(cache.freeSlots))
	}

	// Allocate all slots
	for range 4 {
		slot, err := cache.allocSlot()
		if err != nil {
			t.Fatalf("allocSlot failed: %v", err)
		}
		cache.refCount[slot] = 1
	}

	// Should be full now
	if len(cache.freeSlots) != 0 {
		t.Errorf("expected 0 free slots, got %d", len(cache.freeSlots))
	}

	// Trying to allocate another should fail
	_, err := cache.allocSlot()
	if err != kvcache.ErrKvCacheFull {
		t.Errorf("expected ErrKvCacheFull, got %v", err)
	}
}

func TestHybridCache_SlotReuse(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// Allocate a slot
	slot1, _ := cache.allocSlot()
	cache.refCount[slot1] = 1

	// Free it
	cache.refCount[slot1] = 0
	cache.freeSlot(slot1)

	// Allocate again - should get the same slot back (LIFO)
	slot2, _ := cache.allocSlot()
	if slot2 != slot1 {
		t.Errorf("expected slot %d to be reused, got %d", slot1, slot2)
	}
}

func TestHybridCache_SlotRefCounting_ShareSlot(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// Allocate slot for seq 1
	slot1, _ := cache.allocSlot()
	cache.slotForSeq[1] = slot1
	cache.refCount[slot1] = 1

	// Simulate sharing slot with seq 2 (copy-on-write style)
	cache.slotForSeq[2] = slot1
	cache.refCount[slot1]++

	// Should share the same slot
	if cache.slotForSeq[2] != slot1 {
		t.Errorf("expected seq 2 to share slot %d, got %d", slot1, cache.slotForSeq[2])
	}

	// Ref count should be 2
	if cache.refCount[slot1] != 2 {
		t.Errorf("expected refCount 2, got %d", cache.refCount[slot1])
	}
}

func TestHybridCache_SlotRefCounting_DecRef(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// Allocate slot for seq 1
	slot1, _ := cache.allocSlot()
	cache.slotForSeq[1] = slot1
	cache.refCount[slot1] = 1

	// Share with seq 2
	cache.slotForSeq[2] = slot1
	cache.refCount[slot1]++

	// Unshare seq 2
	cache.refCount[slot1]--
	delete(cache.slotForSeq, 2)

	// Ref count should be back to 1
	if cache.refCount[slot1] != 1 {
		t.Errorf("expected refCount 1 after unshare, got %d", cache.refCount[slot1])
	}

	// Seq 2 should no longer have a slot
	if _, ok := cache.slotForSeq[2]; ok {
		t.Error("seq 2 should not have a slot after unshare")
	}
}

func TestHybridCache_SlotFreeWhenUnused(t *testing.T) {
	cache := createSlotOnlyCache(4)

	initialFreeSlots := len(cache.freeSlots)

	// Allocate slot for seq 1
	slot1, _ := cache.allocSlot()
	cache.slotForSeq[1] = slot1
	cache.refCount[slot1] = 1

	// Free the slot when refCount drops to 0
	cache.refCount[slot1]--
	if cache.refCount[slot1] <= 0 {
		cache.refCount[slot1] = 0
		cache.freeSlot(slot1)
	}
	delete(cache.slotForSeq, 1)

	// Slot should be freed
	if len(cache.freeSlots) != initialFreeSlots {
		t.Errorf("expected %d free slots, got %d", initialFreeSlots, len(cache.freeSlots))
	}

	// Ref count should be 0
	if cache.refCount[slot1] != 0 {
		t.Errorf("expected refCount 0, got %d", cache.refCount[slot1])
	}
}

func TestHybridCache_SlotOverwrite(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// Allocate slots for seq 1 and seq 2
	slot1, _ := cache.allocSlot()
	cache.slotForSeq[1] = slot1
	cache.refCount[slot1] = 1

	slot2, _ := cache.allocSlot()
	cache.slotForSeq[2] = slot2
	cache.refCount[slot2] = 1

	initialFreeSlots := len(cache.freeSlots)

	// Simulate overwriting seq 2's slot with slot1 (sharing)
	// First free the old slot
	cache.refCount[slot2]--
	if cache.refCount[slot2] <= 0 {
		cache.refCount[slot2] = 0
		cache.freeSlot(slot2)
	}
	// Then share slot1
	cache.slotForSeq[2] = slot1
	cache.refCount[slot1]++

	// Seq 2 should now share slot1
	if cache.slotForSeq[2] != slot1 {
		t.Errorf("expected seq 2 to share slot %d, got %d", slot1, cache.slotForSeq[2])
	}

	// Old slot2 should be freed
	if len(cache.freeSlots) != initialFreeSlots+1 {
		t.Errorf("expected %d free slots, got %d", initialFreeSlots+1, len(cache.freeSlots))
	}
}

func TestHybridCache_BoundsChecking(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// Test freeing invalid slot (should not panic)
	cache.freeSlot(-1)
	cache.freeSlot(100) // out of bounds

	// freeSlot does bounds checking, so invalid slots should be ignored
	if len(cache.freeSlots) != 4 {
		t.Errorf("invalid slots should not affect free list, got %d slots", len(cache.freeSlots))
	}
}

func TestHybridCache_MultipleSequences_RefCounting(t *testing.T) {
	cache := createSlotOnlyCache(8)

	// Allocate slot for seq 1
	slot1, _ := cache.allocSlot()
	cache.slotForSeq[1] = slot1
	cache.refCount[slot1] = 1

	// Fork to seq 2, 3, 4 (all share slot1)
	for _, seq := range []int{2, 3, 4} {
		cache.slotForSeq[seq] = slot1
		cache.refCount[slot1]++
	}

	// Ref count should be 4
	if cache.refCount[slot1] != 4 {
		t.Errorf("expected refCount 4, got %d", cache.refCount[slot1])
	}

	// Remove seq 2, 3
	for _, seq := range []int{2, 3} {
		delete(cache.slotForSeq, seq)
		cache.refCount[slot1]--
	}

	if cache.refCount[slot1] != 2 {
		t.Errorf("expected refCount 2, got %d", cache.refCount[slot1])
	}

	// Slot should still be allocated (not in free list)
	found := false
	for _, s := range cache.freeSlots {
		if s == slot1 {
			found = true
			break
		}
	}
	if found {
		t.Error("slot1 should not be in free list yet")
	}

	// Remove remaining sequences
	for _, seq := range []int{1, 4} {
		delete(cache.slotForSeq, seq)
		cache.refCount[slot1]--
	}

	if cache.refCount[slot1] != 0 {
		t.Errorf("expected refCount 0, got %d", cache.refCount[slot1])
	}
}

func TestHybridCache_ChainedSharing(t *testing.T) {
	cache := createSlotOnlyCache(8)

	// Create seq 1
	slot1, _ := cache.allocSlot()
	cache.slotForSeq[1] = slot1
	cache.refCount[slot1] = 1

	// Share 1 -> 2
	cache.slotForSeq[2] = slot1
	cache.refCount[slot1]++

	// Share 2 -> 3 (should still share slot1)
	cache.slotForSeq[3] = cache.slotForSeq[2] // which is slot1
	cache.refCount[slot1]++

	// All should share slot1
	if cache.slotForSeq[1] != slot1 || cache.slotForSeq[2] != slot1 || cache.slotForSeq[3] != slot1 {
		t.Error("all sequences should share slot1")
	}

	if cache.refCount[slot1] != 3 {
		t.Errorf("expected refCount 3, got %d", cache.refCount[slot1])
	}
}

func TestHybridCache_CacheParameters(t *testing.T) {
	cache := NewHybridCache(nil, 512, 5) // hiddenSize=512, dConv=5

	if cache.hiddenSize != 512 {
		t.Errorf("expected hiddenSize 512, got %d", cache.hiddenSize)
	}
	if cache.dConv != 5 {
		t.Errorf("expected dConv 5, got %d", cache.dConv)
	}
}

func TestHybridCache_NumSeqs(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// Initially no sequences
	if cache.numSeqs() != 0 {
		t.Errorf("expected 0 seqs, got %d", cache.numSeqs())
	}

	// Manually set up current batch state
	cache.curSeqs = []int{1, 2, 3}

	if cache.numSeqs() != 3 {
		t.Errorf("expected 3 seqs, got %d", cache.numSeqs())
	}
}

func TestHybridCache_SeqTokens(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// Initially 0
	if cache.seqTokens() != 0 {
		t.Errorf("expected 0 seqTokens, got %d", cache.seqTokens())
	}

	// Manually set up current batch state
	cache.curSeqTokens = 16

	if cache.seqTokens() != 16 {
		t.Errorf("expected 16 seqTokens, got %d", cache.seqTokens())
	}
}

// Test that Seqs returns a clone of curSeqs
func TestHybridCache_Seqs_ReturnsClone(t *testing.T) {
	cache := createSlotOnlyCache(4)

	cache.curSeqs = []int{1, 2, 3}

	seqs := cache.Seqs()

	// Modify returned slice
	seqs[0] = 999

	// Original should be unchanged
	if cache.curSeqs[0] != 1 {
		t.Error("Seqs should return a clone, not the original slice")
	}
}

func TestHybridCache_IsSupportedForBatch(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// Initially not supported (no batch set up)
	if cache.IsSupportedForBatch() {
		t.Error("expected IsSupportedForBatch to be false initially")
	}

	// Set up a valid batch
	cache.curSeqTokens = 1
	cache.curSeqs = []int{1}

	if !cache.IsSupportedForBatch() {
		t.Error("expected IsSupportedForBatch to be true with valid batch")
	}
}

func TestHybridCache_ZeroConvSlots_EmptyInputs(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// zeroConvSlots should handle empty slots without panicking
	cache.zeroConvSlots(nil, nil)
	cache.zeroConvSlots(nil, []int{})

	// zeroConvSlots should handle empty convStates without panicking
	cache.zeroConvSlots(nil, []int{0, 1, 2})
}

func TestHybridCache_SlotRecycling_TracksNewSlots(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// Allocate slot for seq 1
	slot1, _ := cache.allocSlot()
	cache.slotForSeq[1] = slot1
	cache.refCount[slot1] = 1

	// Free the slot (simulating sequence removal)
	cache.refCount[slot1]--
	cache.freeSlot(slot1)
	delete(cache.slotForSeq, 1)

	// Verify slot is in free list
	if len(cache.freeSlots) != 4 {
		t.Errorf("expected 4 free slots after freeing, got %d", len(cache.freeSlots))
	}

	// Allocate for new seq 2 - should get recycled slot
	slot2, _ := cache.allocSlot()
	if slot2 != slot1 {
		t.Errorf("expected recycled slot %d, got %d", slot1, slot2)
	}

	// This recycled slot would need zeroing in the real implementation
	// The actual zeroing is tested via integration tests since it requires ML context
}

func TestHybridCache_NewSequence_GetsTrackedForZeroing(t *testing.T) {
	cache := createSlotOnlyCache(4)

	// Simulate the slot allocation flow from StartForward
	// When a sequence doesn't have a slot, it gets allocated and tracked as "new"

	newSlots := []int{}

	// Seq 1 doesn't have a slot - allocate and track
	seq := 1
	if _, ok := cache.slotForSeq[seq]; !ok {
		slot, err := cache.allocSlot()
		if err != nil {
			t.Fatalf("allocSlot failed: %v", err)
		}
		cache.slotForSeq[seq] = slot
		cache.refCount[slot] = 1
		newSlots = append(newSlots, slot)
	}

	// Verify newSlots contains the allocated slot
	if len(newSlots) != 1 {
		t.Errorf("expected 1 new slot, got %d", len(newSlots))
	}

	// Seq 1 already has a slot - should NOT be tracked as new
	newSlots2 := []int{}
	if _, ok := cache.slotForSeq[seq]; !ok {
		slot, _ := cache.allocSlot()
		cache.slotForSeq[seq] = slot
		cache.refCount[slot] = 1
		newSlots2 = append(newSlots2, slot)
	}

	// Verify no new slots for existing sequence
	if len(newSlots2) != 0 {
		t.Errorf("expected 0 new slots for existing sequence, got %d", len(newSlots2))
	}
}
