package qwen3next

import "testing"

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

func TestHybridCachePrepareRestore(t *testing.T) {
	cache := NewHybridCache(nil, 1, 1, 1)
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
