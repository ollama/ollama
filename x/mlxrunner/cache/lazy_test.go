package cache

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// distinctKV builds a [1, H, L, D] keys/values pair whose values encode the
// absolute token position, so a restored range can be checked against the
// positions it claims. Keys at position p are filled with float(p); values
// with float(-p).
func distinctKV(start, L, H, D int) (*mlx.Array, *mlx.Array) {
	ks := make([]float32, H*L*D)
	vs := make([]float32, H*L*D)
	for h := range H {
		for l := range L {
			for d := range D {
				i := (h*L+l)*D + d
				ks[i] = float32(start + l)
				vs[i] = -float32(start + l)
			}
		}
	}
	return mlx.FromValues(ks, 1, H, L, D), mlx.FromValues(vs, 1, H, L, D)
}

// firstKeyAt returns the keys value stored at sequence position p (channel 0).
func firstKeyAt(arr *mlx.Array, p, D int) float32 {
	return arr.Floats()[p*D]
}

// settledActiveMemory drains unpinned arrays and the allocator cache, then
// reports active (allocated, in-use) bytes.
func settledActiveMemory() int {
	mlx.Sweep()
	mlx.ClearCache()
	return mlx.ActiveMemory()
}

// TestKVSpeculationCaptureAllocatesNothing verifies that an MTP-style capture —
// schedule per-token offsets, run one batched write, take the snapshots, rewind
// via Restore(nil), and Close — allocates no snapshot buffers, because every
// snapshot stays lazy and is discarded before any overwrite. Compare against the
// bytes an eager per-token copy would cost.
func TestKVSpeculationCaptureAllocatesNothing(t *testing.T) {
	skipIfNoMLX(t)

	const before, draft, H, D = 16, 8, 4, 8

	c := NewKVCache()
	fillKV(c, before)

	offsets := make([]int, draft)
	for i := range offsets {
		offsets[i] = before + i
	}
	c.PrepareSnapshots(offsets)

	k, v := batchKV(draft)
	c.Update(newKVBatch(before, draft), k, v)

	baseline := settledActiveMemory()

	snaps := c.TakeSnapshots()
	// Every captured snapshot is a lazy snapshot (no owned buffer).
	for i, s := range snaps {
		if s == nil {
			continue
		}
		if ks := s.(*kvSnapshot); ks.keys != nil {
			t.Fatalf("snaps[%d] owns a buffer at capture; want a lazy snapshot", i)
		}
	}

	// MTP commit: rewind to a partial accept, then discard all snapshots.
	if !c.Restore(nil, before+draft/2) {
		t.Fatal("live rewind failed")
	}
	for _, s := range snaps {
		if s != nil {
			s.Close()
		}
	}

	after := settledActiveMemory()
	// Lazy snapshots allocate nothing; allow a tiny slack for allocator noise but
	// well under one per-token copy (draft tokens * keys+values).
	perToken := (c.keys.NumBytes() + c.values.NumBytes()) / c.keys.Dim(2)
	if after-baseline > perToken {
		t.Fatalf("capture allocated %d bytes (> one token %d); lazy snapshots should allocate nothing",
			after-baseline, perToken)
	}
}

// TestKVLazySnapshotSizeZeroUntilMaterialized verifies the accounting contract:
// a lazy snapshot reports Size() == 0 (it owns no extra memory), and once a
// destructive write triggers copyOut the materialize hook fires with the
// newly-allocated bytes and Size() reports the owned arrays.
func TestKVLazySnapshotSizeZeroUntilMaterialized(t *testing.T) {
	skipIfNoMLX(t)

	const H, D = 4, 8

	c := NewKVCache()
	k, v := distinctKV(0, 10, H, D)
	c.Update(newKVBatch(0, 10), k, v)

	snap := c.Snapshot(5).(*kvSnapshot)
	defer snap.Close()
	if snap.Size() != 0 {
		t.Fatalf("lazy snapshot Size = %d, want 0", snap.Size())
	}

	var hookDelta int
	snap.SetMaterializeHook(func(delta int) { hookDelta = delta })

	// Rewind and overwrite to force copyOut.
	if !c.Restore(nil, 5) {
		t.Fatal("rewind failed")
	}
	nk, nv := distinctKV(100, 3, H, D)
	c.Update(newKVBatch(5, 3), nk, nv)

	if snap.keys == nil {
		t.Fatal("snapshot was not materialized by overwriting write")
	}
	want := snap.keys.NumBytes() + snap.values.NumBytes()
	if hookDelta != want {
		t.Fatalf("hook fired with delta %d, want %d (owned bytes)", hookDelta, want)
	}
	if snap.Size() != want {
		t.Fatalf("materialized Size = %d, want %d", snap.Size(), want)
	}
}

// TestKVLazySnapshotCopiedOutOnOverwrite verifies that after a rewind, a write
// that overwrites a lazy snapshot's slots copies the snapshot out first, so it
// still reads the pre-overwrite data — both keys and values, across the whole
// captured range (guarding the Slice+Contiguous copy-out representation).
func TestKVLazySnapshotCopiedOutOnOverwrite(t *testing.T) {
	skipIfNoMLX(t)

	const H, D = 4, 8

	c := NewKVCache()
	// Fill [0,10) with position-encoded values (keys p, values -p).
	k, v := distinctKV(0, 10, H, D)
	c.Update(newKVBatch(0, 10), k, v)

	// Lazy snapshot [5,10).
	snap := c.Snapshot(5).(*kvSnapshot)
	if snap.keys != nil {
		t.Fatal("Snapshot should return a lazy snapshot")
	}

	// Rewind to 5 and overwrite [5,8) with different data.
	if !c.Restore(nil, 5) {
		t.Fatal("rewind failed")
	}
	nk, nv := distinctKV(100, 3, H, D) // positions 100,101,102
	c.Update(newKVBatch(5, 3), nk, nv)

	// The overwrite must have copied the lazy snapshot out beforehand, preserving
	// the pre-overwrite keys and values across the whole [5,10) range.
	if snap.keys == nil {
		t.Fatal("lazy snapshot was not copied out before the overwriting write")
	}
	mlx.Eval(snap.keys, snap.values)
	keys := snap.keys.Floats()
	vals := snap.values.Floats()
	for l := range snap.toOffset - snap.fromOffset {
		wantK := float32(snap.fromOffset + l)
		if got := keys[l*D]; got != wantK {
			t.Fatalf("snapshot keys[%d] = %v, want %v (pre-overwrite data)", l, got, wantK)
		}
		if got := vals[l*D]; got != -wantK {
			t.Fatalf("snapshot values[%d] = %v, want %v (pre-overwrite data)", l, got, -wantK)
		}
	}
	snap.Close()
}

// TestKVLazySnapshotCopiedOutOnFree verifies Free copies out outstanding lazy snapshots
// so they survive after the cache buffer is gone.
func TestKVLazySnapshotCopiedOutOnFree(t *testing.T) {
	skipIfNoMLX(t)

	const H, D = 4, 8

	c := NewKVCache()
	k, v := distinctKV(0, 10, H, D)
	c.Update(newKVBatch(0, 10), k, v)

	snap := c.Snapshot(5).(*kvSnapshot)
	c.Free()

	if snap.keys == nil {
		t.Fatal("lazy snapshot was not copied out on Free")
	}
	mlx.Eval(snap.keys)
	if got := firstKeyAt(snap.keys, 0, D); got != 5 {
		t.Fatalf("snapshot[0] key = %v, want 5 (data preserved through Free)", got)
	}
	snap.Close()
}

// TestKVLazySnapshotSplitMergeNoCopy verifies Split of a lazy snapshot and Merge of two
// adjacent lazy snapshots are pure arithmetic — they produce lazy snapshots and allocate
// nothing — while still tracking the correct offsets and data.
func TestKVLazySnapshotSplitMergeNoCopy(t *testing.T) {
	skipIfNoMLX(t)

	const H, D = 4, 8

	c := NewKVCache()
	k, v := distinctKV(0, 10, H, D)
	c.Update(newKVBatch(0, 10), k, v)

	base := settledActiveMemory()

	// Lazy snapshot [2,10), split at 5.
	snap := c.Snapshot(2)
	p, ch := c.Split(snap, 5)
	ps, cs := p.(*kvSnapshot), ch.(*kvSnapshot)
	if ps.keys != nil || cs.keys != nil {
		t.Fatal("Split of a lazy snapshot should yield lazy snapshots (no copy)")
	}
	if ps.fromOffset != 2 || ps.toOffset != 5 || cs.fromOffset != 5 || cs.toOffset != 10 {
		t.Fatalf("split ranges = [%d,%d)/[%d,%d), want [2,5)/[5,10)", ps.fromOffset, ps.toOffset, cs.fromOffset, cs.toOffset)
	}

	// Merge them back into [2,10).
	merged := c.Merge(p, ch).(*kvSnapshot)
	if merged.keys != nil {
		t.Fatal("Merge of adjacent lazy snapshots should yield a lazy snapshot (no Concatenate)")
	}
	if merged.fromOffset != 2 || merged.toOffset != 10 {
		t.Fatalf("merged range = [%d,%d), want [2,10)", merged.fromOffset, merged.toOffset)
	}

	if after := settledActiveMemory(); after > base {
		t.Fatalf("Split/Merge of lazy snapshots allocated %d bytes; want 0", after-base)
	}
	merged.Close()
}

// TestKVLazySnapshotSurvivesPathSwitch reproduces the switchToPath sequence that
// would otherwise corrupt a trie-held lazy snapshot: page out the diverging leaf
// (Snapshot), rewind to the common ancestor (Restore(nil)), then page in a
// different path (Restore feeds new data via appendKV, overwriting the old
// leaf's slots). The paged-out snapshot must still hold the original tokens.
func TestKVLazySnapshotSurvivesPathSwitch(t *testing.T) {
	skipIfNoMLX(t)

	const H, D = 4, 8

	c := NewKVCache()
	// Active path tokens [0,10): positions 0..9.
	k, v := distinctKV(0, 10, H, D)
	c.Update(newKVBatch(0, 10), k, v)

	// Page out the diverging leaf [4,10) as a lazy snapshot (what switchToPath does
	// before rewinding).
	leaf := c.Snapshot(4).(*kvSnapshot)

	// Rewind to the common ancestor at 4 (Restore(nil): offset move only).
	if !c.Restore(nil, 4) {
		t.Fatal("rewind to ancestor failed")
	}

	// Page in the new path [4,9): positions 200..204, overwriting the old leaf's
	// slots. appendKV must copy the leaf lazy snapshot out first.
	nk, nv := distinctKV(200, 5, H, D)
	c.Update(newKVBatch(4, 5), nk, nv)

	if leaf.keys == nil {
		t.Fatal("leaf lazy snapshot was not copied out during page-in")
	}
	mlx.Eval(leaf.keys)
	// leaf covers [4,10): its keys are the original positions 4..9.
	keys := leaf.keys.Floats()
	for l := range leaf.toOffset - leaf.fromOffset {
		if got, want := keys[l*D], float32(leaf.fromOffset+l); got != want {
			t.Fatalf("paged-out leaf key[%d] = %v, want %v (original path data)", l, got, want)
		}
	}
	leaf.Close()
}

// TestKVRestoreLiveLazySnapshotIsOffsetMove verifies the same-path rewind/rematch
// fast path: restoring a snapshot that is still a lazy index into this cache's
// own buffer (its slots never overwritten, so the data is already live) advances
// the offset without cloning or replaying. This is the switchToPath sequence
// where a paged-out leaf is restored before any write displaced it.
func TestKVRestoreLiveLazySnapshotIsOffsetMove(t *testing.T) {
	skipIfNoMLX(t)

	const H, D = 4, 8

	c := NewKVCache()
	k, v := distinctKV(0, 10, H, D)
	c.Update(newKVBatch(0, 10), k, v)

	// Page out the leaf [5,10) as a lazy snapshot, then rewind (offset move only).
	snap := c.Snapshot(5).(*kvSnapshot)
	if !c.Restore(nil, 5) {
		t.Fatal("rewind failed")
	}

	base := settledActiveMemory()

	// Restore the snapshot back to 10. Its slots [5,10) were never overwritten,
	// so it is still lazy and the data is already in the buffer — a pure offset
	// move, no allocation.
	if !c.Restore(snap, 10) {
		t.Fatal("restore failed")
	}
	if snap.keys != nil {
		t.Fatal("snapshot was copied out; expected the offset-move fast path")
	}
	if c.Offset() != 10 {
		t.Fatalf("offset after restore = %d, want 10", c.Offset())
	}
	if after := settledActiveMemory(); after > base {
		t.Fatalf("restore of a live lazy snapshot allocated %d bytes; want 0 (offset move)", after-base)
	}

	// The buffer still holds the original positions 0..9.
	st := c.State()
	mlx.Eval(st[0])
	keys := st[0].Floats()
	for l := range 10 {
		if got := keys[l*D]; got != float32(l) {
			t.Fatalf("restored key[%d] = %v, want %v", l, got, float32(l))
		}
	}
	snap.Close()
}
