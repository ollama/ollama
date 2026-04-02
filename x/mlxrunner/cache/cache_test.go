package cache

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

var singleTokenBatch = &batch.ForwardBatch{SeqIDs: []int{0}, SeqLens: []int{1}}

func newKVCacheWithSeq() *KVCache {
	c := NewKVCache()
	c.SetSeqs([]int{0})
	return c
}

func newRotatingKVCacheWithSeq(maxSize int) *RotatingKVCache {
	c := NewRotatingKVCache(maxSize)
	c.SetSeqs([]int{0})
	return c
}

func TestKVCacheSnapshotRestoreNeedBase(t *testing.T) {
	skipIfNoMLX(t)
	c := newKVCacheWithSeq()

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(singleTokenBatch, k, v)
	}

	snap := c.Snapshot(0, 5)

	c.Free()

	if c.Restore(0, snap, 10) {
		t.Fatal("expected Restore to fail with no base data")
	}
}

func TestKVCacheDataSurvivesSnapshotRestore(t *testing.T) {
	skipIfNoMLX(t)
	c := newKVCacheWithSeq()

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(singleTokenBatch, k, v)
	}

	snap := c.Snapshot(0, 0)
	if snap == nil {
		t.Fatal("Snapshot returned nil")
	}

	c2 := newKVCacheWithSeq()
	if !c2.Restore(0, snap, 10) {
		t.Fatal("Restore failed")
	}
	if int(c2.Offsets(0)[0]) != 10 {
		t.Fatalf("offset = %d, want 10", int(c2.Offsets(0)[0]))
	}

	state := c2.State()
	if len(state) != 2 {
		t.Fatalf("State() returned %d arrays, want 2", len(state))
	}
	if state[0].Dim(2) < 10 {
		t.Fatalf("keys seq dim = %d, want >= 10", state[0].Dim(2))
	}
}

func TestKVCacheSplitPreservesData(t *testing.T) {
	skipIfNoMLX(t)
	c := newKVCacheWithSeq()

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(singleTokenBatch, k, v)
	}

	snap := c.Snapshot(0, 0)
	parent, child := c.Split(snap, 5)
	if parent == nil || child == nil {
		t.Fatal("Split returned nil")
	}

	c2 := newKVCacheWithSeq()
	if !c2.Restore(0, parent, 5) {
		t.Fatal("Restore(parent) failed")
	}
	if int(c2.Offsets(0)[0]) != 5 {
		t.Fatalf("offset after parent = %d, want 5", int(c2.Offsets(0)[0]))
	}

	if !c2.Restore(0, child, 10) {
		t.Fatal("Restore(child) failed")
	}
	if int(c2.Offsets(0)[0]) != 10 {
		t.Fatalf("offset after child = %d, want 10", int(c2.Offsets(0)[0]))
	}
}

func TestKVCacheSplitMergeRoundTripData(t *testing.T) {
	skipIfNoMLX(t)
	c := newKVCacheWithSeq()

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(singleTokenBatch, k, v)
	}

	snap := c.Snapshot(0, 0)
	parent, child := c.Split(snap, 6)
	merged := c.Merge(parent, child)
	if merged == nil {
		t.Fatal("Merge returned nil")
	}

	c2 := newKVCacheWithSeq()
	if !c2.Restore(0, merged, 10) {
		t.Fatal("Restore(merged) failed")
	}
	if int(c2.Offsets(0)[0]) != 10 {
		t.Fatalf("offset = %d, want 10", int(c2.Offsets(0)[0]))
	}
}

func TestRotatingKVCacheRewindOutsideWindow(t *testing.T) {
	skipIfNoMLX(t)
	c := newRotatingKVCacheWithSeq(4)

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(singleTokenBatch, k, v)
	}

	if c.Restore(0, nil, 3) {
		t.Fatal("Restore(nil, 3) should fail when outside window")
	}
}

func TestRotatingKVCacheWindowedHistory(t *testing.T) {
	skipIfNoMLX(t)
	c := newRotatingKVCacheWithSeq(4)

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(singleTokenBatch, k, v)
	}

	k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
	v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
	_, _, kv := c.Update(singleTokenBatch, k, v)

	if len(kv.SeqLens) != 1 {
		t.Fatalf("SeqLens length = %d, want 1", len(kv.SeqLens))
	}
	if kv.SeqLens[0] != 4 {
		t.Fatalf("SeqLens[0] = %d, want 4 (window size)", kv.SeqLens[0])
	}
}

func TestRotatingKVCacheRestoreFromSnapshot(t *testing.T) {
	skipIfNoMLX(t)
	c := newRotatingKVCacheWithSeq(8)

	for range 3 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(singleTokenBatch, k, v)
	}
	if int(c.Offsets(0)[0]) != 3 {
		t.Fatalf("offset = %d, want 3", int(c.Offsets(0)[0]))
	}

	// Rewind before wrap should succeed
	if !c.Restore(0, nil, 1) {
		t.Fatal("Restore(nil, 1) should succeed before wrap")
	}
	if int(c.Offsets(0)[0]) != 1 {
		t.Fatalf("offset after restore = %d, want 1", int(c.Offsets(0)[0]))
	}
}

func TestKVCacheMultiSeqUpdate(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()
	c.SetSeqs([]int{0, 1})

	// Prefill: seq 0 gets 3 tokens, seq 1 gets 5 tokens
	b := &batch.ForwardBatch{SeqIDs: []int{0, 1}, SeqLens: []int{3, 5}}
	k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 8, 8)
	v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 8, 8)
	c.Update(b, k, v)

	if int(c.Offsets(0)[0]) != 3 {
		t.Fatalf("seq 0 offset = %d, want 3", int(c.Offsets(0)[0]))
	}
	if int(c.Offsets(1)[0]) != 5 {
		t.Fatalf("seq 1 offset = %d, want 5", int(c.Offsets(1)[0]))
	}

	// Decode: each seq gets 1 token
	b2 := &batch.ForwardBatch{SeqIDs: []int{0, 1}, SeqLens: []int{1, 1}}
	k2 := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 2, 8)
	v2 := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 2, 8)
	c.Update(b2, k2, v2)

	if int(c.Offsets(0)[0]) != 4 {
		t.Fatalf("seq 0 offset after decode = %d, want 4", int(c.Offsets(0)[0]))
	}
	if int(c.Offsets(1)[0]) != 6 {
		t.Fatalf("seq 1 offset after decode = %d, want 6", int(c.Offsets(1)[0]))
	}
}

func TestKVCacheSetSeqsAndUpdate(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()
	c.SetSeqs([]int{0, 1})

	b := &batch.ForwardBatch{SeqIDs: []int{0, 1}, SeqLens: []int{3, 3}}
	k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 6, 8)
	v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 6, 8)
	c.Update(b, k, v)

	c.SetSeqs([]int{1})

	// Update surviving sequence
	b2 := &batch.ForwardBatch{SeqIDs: []int{1}, SeqLens: []int{1}}
	k2 := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
	v2 := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
	c.Update(b2, k2, v2)

	if int(c.Offsets(1)[0]) != 4 {
		t.Fatalf("seq 1 offset = %d, want 4", int(c.Offsets(1)[0]))
	}
}

func TestKVCacheRebuildWithOldLengths(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()
	c.SetSeqs([]int{0})

	// Fill to capacity boundary
	for range 256 {
		b := singleTokenBatch
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(b, k, v)
	}

	if int(c.Offsets(0)[0]) != 256 {
		t.Fatalf("offset = %d, want 256", int(c.Offsets(0)[0]))
	}

	// Next token triggers rebuild (exceeds capacity)
	k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
	v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
	c.Update(singleTokenBatch, k, v)

	if int(c.Offsets(0)[0]) != 257 {
		t.Fatalf("offset after rebuild = %d, want 257", int(c.Offsets(0)[0]))
	}
}

func TestRotatingKVCacheMultiSeqWindowedHistory(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRotatingKVCache(4)
	c.SetSeqs([]int{0, 1})

	// Fill both sequences past the window
	for range 6 {
		b := &batch.ForwardBatch{SeqIDs: []int{0, 1}, SeqLens: []int{1, 1}}
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 2, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 2, 8)
		_, _, kv := c.Update(b, k, v)

		// After enough tokens, SeqLens should be clamped to window
		if kv.SeqLens[0] > 4 || kv.SeqLens[1] > 4 {
			t.Fatalf("SeqLens %v exceed window size 4", kv.SeqLens)
		}
	}

	offsets := c.Offsets(0, 1)
	if int(offsets[0]) != 6 || int(offsets[1]) != 6 {
		t.Fatalf("offsets = %v, want [6 6]", offsets)
	}
}

func TestKVCacheSetSeqsAfterMaterialized(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()
	c.SetSeqs([]int{0})

	// Materialize with some tokens
	for range 5 {
		b := singleTokenBatch
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(b, k, v)
	}
	if int(c.Offsets(0)[0]) != 5 {
		t.Fatalf("seq 0 offset = %d, want 5", int(c.Offsets(0)[0]))
	}

	// Add a new sequence after buffer already exists
	c.SetSeqs([]int{0, 1})

	// Update both sequences
	b := &batch.ForwardBatch{SeqIDs: []int{0, 1}, SeqLens: []int{1, 1}}
	k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 2, 8)
	v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 2, 8)
	c.Update(b, k, v)

	if int(c.Offsets(0)[0]) != 6 {
		t.Fatalf("seq 0 offset = %d, want 6", int(c.Offsets(0)[0]))
	}
	if int(c.Offsets(1)[0]) != 1 {
		t.Fatalf("seq 1 offset = %d, want 1", int(c.Offsets(1)[0]))
	}
}

func TestRotatingKVCacheSetSeqsAfterMaterialized(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRotatingKVCache(4)
	c.SetSeqs([]int{0})

	for range 3 {
		b := singleTokenBatch
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(b, k, v)
	}

	// Add after materialized
	c.SetSeqs([]int{0, 1})

	b := &batch.ForwardBatch{SeqIDs: []int{0, 1}, SeqLens: []int{1, 1}}
	k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 2, 8)
	v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 2, 8)
	c.Update(b, k, v)

	if int(c.Offsets(0)[0]) != 4 {
		t.Fatalf("seq 0 offset = %d, want 4", int(c.Offsets(0)[0]))
	}
	if int(c.Offsets(1)[0]) != 1 {
		t.Fatalf("seq 1 offset = %d, want 1", int(c.Offsets(1)[0]))
	}
}
