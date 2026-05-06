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

// newKVBatch builds a B=1 batch at SeqOffsets=off with all-real
// queries (SeqQueryLens=L) — the standard single-sequence cache
// test shape.
func newKVBatch(off, L int) *batch.Batch {
	return &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, L),
		SeqOffsets:   []int32{int32(off)},
		SeqQueryLens: []int32{int32(L)},
	}
}

func TestKVCacheSnapshotRestoreNeedBase(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	}

	// Snapshot [5, 10).
	snap := c.Snapshot(5)

	// Free the cache completely — offset is now 0.
	c.Free()

	// Restore should fail because cache doesn't have data up to fromOffset=5.
	if c.Restore(snap, 10) {
		t.Fatal("expected Restore to fail with no base data")
	}
}

// TestKVCacheDataSurvivesSnapshotRestore verifies that actual array data
// is preserved through a snapshot→free→restore cycle.
func TestKVCacheDataSurvivesSnapshotRestore(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	}

	snap := c.Snapshot(0)
	if snap == nil {
		t.Fatal("Snapshot returned nil")
	}

	// Free and restore to a fresh cache.
	c2 := NewKVCache()
	if !c2.Restore(snap, 10) {
		t.Fatal("Restore failed")
	}
	if c2.Offset() != 10 {
		t.Fatalf("offset = %d, want 10", c2.Offset())
	}

	// Verify State() returns arrays with correct sequence dimension.
	state := c2.State()
	if len(state) != 2 {
		t.Fatalf("State() returned %d arrays, want 2", len(state))
	}
	// keys shape: [B, H, seqLen, Dk]
	if state[0].Dim(2) != 10 {
		t.Fatalf("keys seq dim = %d, want 10", state[0].Dim(2))
	}
	if state[1].Dim(2) != 10 {
		t.Fatalf("values seq dim = %d, want 10", state[1].Dim(2))
	}
}

// TestKVCacheSplitPreservesData verifies that split produces two snapshots
// that can be sequentially restored to rebuild the original cache state.
func TestKVCacheSplitPreservesData(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	}

	snap := c.Snapshot(0)
	parent, child := c.Split(snap, 5)
	if parent == nil || child == nil {
		t.Fatal("Split returned nil")
	}

	// Restore parent → offset=5, seq dim=5.
	c2 := NewKVCache()
	if !c2.Restore(parent, 5) {
		t.Fatal("Restore(parent) failed")
	}
	if c2.Offset() != 5 {
		t.Fatalf("offset after parent = %d, want 5", c2.Offset())
	}
	state := c2.State()
	if state[0].Dim(2) != 5 {
		t.Fatalf("keys seq dim after parent = %d, want 5", state[0].Dim(2))
	}

	// Restore child on top → offset=10, seq dim=10.
	if !c2.Restore(child, 10) {
		t.Fatal("Restore(child) failed")
	}
	if c2.Offset() != 10 {
		t.Fatalf("offset after child = %d, want 10", c2.Offset())
	}
	state = c2.State()
	if state[0].Dim(2) != 10 {
		t.Fatalf("keys seq dim after child = %d, want 10", state[0].Dim(2))
	}
}

// TestKVCacheSplitMergeRoundTripData verifies that splitting and merging back
// produces a snapshot equivalent to the original.
func TestKVCacheSplitMergeRoundTripData(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	}

	snap := c.Snapshot(0)
	parent, child := c.Split(snap, 6)
	merged := c.Merge(parent, child)
	if merged == nil {
		t.Fatal("Merge returned nil")
	}

	c2 := NewKVCache()
	if !c2.Restore(merged, 10) {
		t.Fatal("Restore(merged) failed")
	}
	if c2.Offset() != 10 {
		t.Fatalf("offset = %d, want 10", c2.Offset())
	}

	state := c2.State()
	if state[0].Dim(2) != 10 {
		t.Fatalf("keys seq dim = %d, want 10", state[0].Dim(2))
	}
	if state[1].Dim(2) != 10 {
		t.Fatalf("values seq dim = %d, want 10", state[1].Dim(2))
	}
}

func TestRotatingKVCacheRestoreOutsideWindow(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRotatingKVCache(4)

	// Feed 10 tokens (window size 4, so positions 0-5 are evicted).
	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	}

	// Offset 3 is outside the window.
	if c.Restore(nil, 3) {
		t.Fatal("Restore(nil, 3) should fail when outside window")
	}
}

// TestRotatingKVCacheSnapshotPreservesWindow verifies that after restoring
// from a snapshot, the rotating cache has the correct window of data.
func TestRotatingKVCacheSnapshotPreservesWindow(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRotatingKVCache(4)

	// Feed 10 tokens one at a time. Window size 4, so only last 4 are kept.
	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	}

	snap := c.Snapshot(0)
	if snap == nil {
		t.Fatal("Snapshot returned nil")
	}

	// Feed 5 more tokens.
	for range 5 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	}

	// Restore to offset 10.
	if !c.Restore(snap, 10) {
		t.Fatal("Restore failed")
	}
	if c.Offset() != 10 {
		t.Fatalf("offset = %d, want 10", c.Offset())
	}

	state := c.State()
	if len(state) != 2 {
		t.Fatalf("State() returned %d arrays, want 2", len(state))
	}
	// Seq dim should be min(offset, maxSize) = min(10, 4) = 4.
	seqDim := state[0].Dim(2)
	if seqDim != 4 {
		t.Fatalf("keys seq dim = %d, want 4 (window size)", seqDim)
	}
}

// TestRotatingKVCacheRestoreFromSnapshot verifies that restoring from a
// snapshot correctly preserves the write position (idx), so subsequent
// single-token updates land in the right buffer slot.
func TestRotatingKVCacheRestoreFromSnapshot(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRotatingKVCache(4)

	// Fill the window: 6 tokens into a size-4 window.
	// After this, idx has wrapped and the buffer has rotated.
	for range 6 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	}
	if c.Offset() != 6 {
		t.Fatalf("offset = %d, want 6", c.Offset())
	}

	snap := c.Snapshot(0)

	// Mutate the cache further so live state diverges from snapshot.
	for range 3 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	}

	// Restore to snapshot state.
	if !c.Restore(snap, 6) {
		t.Fatal("Restore failed")
	}
	if c.Offset() != 6 {
		t.Fatalf("offset after restore = %d, want 6", c.Offset())
	}

	// Feed one more token. If idx was restored correctly, this should
	// produce a valid window of size 4 at offset 7.
	k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
	v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
	c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)

	if c.Offset() != 7 {
		t.Fatalf("offset after post-restore update = %d, want 7", c.Offset())
	}
	state := c.State()
	if len(state) != 2 {
		t.Fatalf("State() returned %d arrays, want 2", len(state))
	}
	seqDim := state[0].Dim(2)
	if seqDim != 4 {
		t.Fatalf("keys seq dim = %d, want 4 (window size)", seqDim)
	}
}
