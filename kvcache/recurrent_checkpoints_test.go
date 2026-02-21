package kvcache

import (
	"errors"
	"math"
	"testing"

	"github.com/ollama/ollama/ml"
)

func newTestCache() *Recurrent {
	return NewRecurrentCache(RecurrentConfig{ConvDim: 1, ConvChannels: 2, RecurrentStateSize: 2})
}

func TestSlotCheckpointStoreBestIndex(t *testing.T) {
	store := newSlotCheckpointStore(2)
	store.record(10)
	store.record(20)

	_, pos, ok := store.bestIndex(15)
	if !ok || pos != 10 {
		t.Fatalf("expected best pos 10, got pos=%d ok=%v", pos, ok)
	}

	store.record(30) // overwrite oldest (10)

	if _, _, ok := store.bestIndex(15); ok {
		t.Fatalf("expected no checkpoint for targetPos=15 after overwrite")
	}

	_, pos, ok = store.bestIndex(40)
	if !ok || pos != 30 {
		t.Fatalf("expected best pos 30, got pos=%d ok=%v", pos, ok)
	}
}

func TestCachePrepareRestore(t *testing.T) {
	cache := newTestCache()
	cache.checkpointCount = 3
	cache.checkpoints = make(map[int]*slotCheckpointStore)
	cache.pendingRestore = make(map[int]checkpointRestore)

	cache.slotForSeq[1] = 0
	store := cache.checkpointStore(0)
	store.record(5)
	store.record(9)
	store.record(15)

	restorePos, ok := cache.PrepareRestore(1, 12)
	if !ok {
		t.Fatalf("expected restore ok")
	}
	if restorePos != 10 {
		t.Fatalf("expected restorePos 10, got %d", restorePos)
	}
	rest, ok := cache.pendingRestore[1]
	if !ok {
		t.Fatalf("expected pending restore entry")
	}
	if rest.pos != 9 {
		t.Fatalf("expected pending restore pos 9, got %d", rest.pos)
	}
}

func TestSlotCheckpointStorePruneAfter(t *testing.T) {
	store := newSlotCheckpointStore(3)
	store.record(10)
	store.record(20)
	store.record(30)

	store.pruneAfter(20)

	if store.lastPos != 20 {
		t.Fatalf("expected lastPos 20, got %d", store.lastPos)
	}

	_, pos, ok := store.bestIndex(25)
	if !ok || pos != 20 {
		t.Fatalf("expected best pos 20 after prune, got pos=%d ok=%v", pos, ok)
	}

	_, pos, ok = store.bestIndex(35)
	if !ok || pos != 20 {
		t.Fatalf("expected pruned best pos 20 for targetPos=35, got pos=%d ok=%v", pos, ok)
	}
}

func TestCacheRestoreRejectsIncompleteCheckpoint(t *testing.T) {
	cache := newTestCache()
	cache.checkpointCount = 3
	cache.checkpoints = make(map[int]*slotCheckpointStore)
	cache.pendingRestore = make(map[int]checkpointRestore)

	cache.slotForSeq[1] = 0
	cache.refCount = []int{1}
	cache.freeSlots = nil

	// Simulate layer 0 requires both conv and recurrent checkpoints.
	cache.convStates[0] = nil
	cache.recurrentStates[0] = nil

	store := cache.checkpointStore(0)
	idx := store.record(9)
	entry := &store.entries[idx]
	entry.conv = map[int]ml.Tensor{0: nil}
	// entry.recurrent intentionally missing

	cache.pendingRestore[1] = checkpointRestore{slot: 0, idx: idx, pos: 9}

	err := cache.Remove(1, 10, math.MaxInt32)
	if !errors.Is(err, ErrNotSupported) {
		t.Fatalf("expected ErrNotSupported for incomplete checkpoint, got %v", err)
	}
}

func TestCacheRestoreAcceptsCompleteCheckpoint(t *testing.T) {
	cache := newTestCache()
	cache.checkpointCount = 3
	cache.checkpoints = make(map[int]*slotCheckpointStore)
	cache.pendingRestore = make(map[int]checkpointRestore)

	cache.slotForSeq[1] = 0
	cache.refCount = []int{1}
	cache.freeSlots = nil

	store := cache.checkpointStore(0)
	idx := store.record(9)

	cache.pendingRestore[1] = checkpointRestore{slot: 0, idx: idx, pos: 9}

	restore := cache.pendingRestore[1]
	if !cache.restoreComplete(restore) {
		t.Fatalf("expected restoreComplete to return true for complete checkpoint")
	}
}

func TestCacheRecurrentStateShapeValidation(t *testing.T) {
	cache := newTestCache()
	_, err := cache.RecurrentState(nil, 0, 3)
	if !errors.Is(err, ErrInvalidRecurrentShape) {
		t.Fatalf("expected ErrInvalidRecurrentShape, got %v", err)
	}
}

func TestSlotCheckpointStoreRingBufferWrapAround(t *testing.T) {
	store := newSlotCheckpointStore(3)

	store.record(10)
	store.record(20)
	store.record(30)

	store.entries[0].conv = make(map[int]ml.Tensor)
	store.entries[0].conv[0] = nil
	store.entries[0].recurrent = make(map[int]ml.Tensor)
	store.entries[0].recurrent[0] = nil

	store.record(40)

	if store.entries[0].conv == nil {
		t.Fatalf("expected conv map to be preserved on reuse")
	}
	if store.entries[0].recurrent == nil {
		t.Fatalf("expected recurrent map to be preserved on reuse")
	}
	if store.entries[0].pos != 40 {
		t.Fatalf("expected entry 0 pos to be 40, got %d", store.entries[0].pos)
	}
}

func TestSlotCheckpointStoreFullCapacity(t *testing.T) {
	store := newSlotCheckpointStore(2)

	idx1 := store.record(10)
	idx2 := store.record(20)

	if idx1 != 0 || idx2 != 1 {
		t.Fatalf("expected indices 0, 1, got %d, %d", idx1, idx2)
	}
	if store.size != 2 {
		t.Fatalf("expected size 2, got %d", store.size)
	}

	_, pos1, ok1 := store.bestIndex(15)
	_, pos2, ok2 := store.bestIndex(25)

	if !ok1 || pos1 != 10 {
		t.Fatalf("expected best pos 10 for target 15, got pos=%d ok=%v", pos1, ok1)
	}
	if !ok2 || pos2 != 20 {
		t.Fatalf("expected best pos 20 for target 25, got pos=%d ok=%v", pos2, ok2)
	}
}

func TestSlotCheckpointStoreEmptyBuffer(t *testing.T) {
	store := newSlotCheckpointStore(0)

	idx := store.record(10)
	if idx != -1 {
		t.Fatalf("expected record to return -1 for empty buffer, got %d", idx)
	}

	_, _, ok := store.bestIndex(15)
	if ok {
		t.Fatalf("expected no checkpoint for empty buffer")
	}
}

func TestSlotCheckpointStorePruneAfterAll(t *testing.T) {
	store := newSlotCheckpointStore(3)
	store.record(10)
	store.record(20)
	store.record(30)

	store.pruneAfter(5)

	if store.size != 0 {
		t.Fatalf("expected size 0 after pruning all, got %d", store.size)
	}
	if store.lastPos != -1 {
		t.Fatalf("expected lastPos -1 after pruning all, got %d", store.lastPos)
	}

	_, _, ok := store.bestIndex(100)
	if ok {
		t.Fatalf("expected no checkpoint after pruning all")
	}
}
