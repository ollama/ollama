package qwen3next

import (
	"errors"
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

func newTestBackend(tb testing.TB) ml.Backend {
	tb.Helper()

	f, err := os.CreateTemp(tb.TempDir(), "*.gguf")
	if err != nil {
		tb.Fatal(err)
	}
	if err := ggml.WriteGGUF(f, ggml.KV{"general.architecture": "test"}, nil); err != nil {
		_ = f.Close()
		tb.Fatal(err)
	}
	if err := f.Close(); err != nil {
		tb.Fatal(err)
	}

	b, err := ml.NewBackend(f.Name(), ml.BackendParams{AllocMemory: true})
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		b.Close()
	})

	return b
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

func TestHybridCacheRestoreDetachesSharedSlot(t *testing.T) {
	backend := newTestBackend(t)

	cache := NewHybridCache(nil, 1, 2, 2)
	cache.Init(backend, ml.DTypeF16, 2, 8, 2)

	cache.slotForSeq[1] = 0
	cache.slotForSeq[2] = 0
	cache.refCount[0] = 2
	cache.refCount[1] = 0
	cache.freeSlots = []int{1}

	store := cache.checkpointStore(0)
	idx := store.record(9)
	cache.pendingRestore[1] = checkpointRestore{slot: 0, idx: idx, pos: 9}

	if err := cache.Remove(1, 10, math.MaxInt32); err != nil {
		t.Fatalf("Remove failed: %v", err)
	}

	if cache.slotForSeq[1] == cache.slotForSeq[2] {
		t.Fatalf("expected restore to detach shared slot, got same slot %d", cache.slotForSeq[1])
	}
	if cache.slotForSeq[1] != 1 {
		t.Fatalf("expected seq 1 to move to slot 1, got %d", cache.slotForSeq[1])
	}
	if cache.slotForSeq[2] != 0 {
		t.Fatalf("expected seq 2 to remain on slot 0, got %d", cache.slotForSeq[2])
	}
	if cache.refCount[0] != 1 || cache.refCount[1] != 1 {
		t.Fatalf("unexpected refCounts: slot0=%d slot1=%d", cache.refCount[0], cache.refCount[1])
	}
	if _, ok := cache.pendingRestore[1]; ok {
		t.Fatalf("expected pending restore to be cleared")
	}
}

func TestHybridCacheRestoreRejectsIncompleteCheckpoint(t *testing.T) {
	backend := newTestBackend(t)

	cache := NewHybridCache(nil, 1, 2, 2)
	cache.Init(backend, ml.DTypeF16, 1, 8, 2)

	cache.slotForSeq[1] = 0
	cache.refCount[0] = 1

	ctx := backend.NewContext()
	defer ctx.Close()

	_ = cache.convBuffer(ctx, 0)
	_ = cache.deltaBuffer(ctx, 0)

	store := cache.checkpointStore(0)
	idx := store.record(9)
	entry := &store.entries[idx]
	_ = cache.ensureCheckpointConv(0, entry)

	cache.pendingRestore[1] = checkpointRestore{slot: 0, idx: idx, pos: 9}

	err := cache.Remove(1, 10, math.MaxInt32)
	if !errors.Is(err, kvcache.ErrNotSupported) {
		t.Fatalf("expected ErrNotSupported for incomplete checkpoint, got %v", err)
	}
}

func TestHybridCacheRestoreAcceptsCompleteCheckpoint(t *testing.T) {
	backend := newTestBackend(t)

	cache := NewHybridCache(nil, 1, 2, 2)
	cache.Init(backend, ml.DTypeF16, 1, 8, 2)

	cache.slotForSeq[1] = 0
	cache.refCount[0] = 1

	ctx := backend.NewContext()
	defer ctx.Close()

	_ = cache.convBuffer(ctx, 0)
	_ = cache.deltaBuffer(ctx, 0)

	store := cache.checkpointStore(0)
	idx := store.record(9)
	entry := &store.entries[idx]
	_ = cache.ensureCheckpointConv(0, entry)
	_ = cache.ensureCheckpointDelta(0, entry)

	cache.pendingRestore[1] = checkpointRestore{slot: 0, idx: idx, pos: 9}

	if err := cache.Remove(1, 10, math.MaxInt32); err != nil {
		t.Fatalf("expected restore to succeed, got %v", err)
	}
}
