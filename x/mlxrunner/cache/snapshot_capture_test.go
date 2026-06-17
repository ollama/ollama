package cache

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// fillKV writes n single-token steps into c, one Update per token, so the
// cache reaches offset n the way decode would.
func fillKV(c Attention, n int) {
	for range n {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	}
}

// batchKV builds K/V for an L-token batched write.
func batchKV(L int) (*mlx.Array, *mlx.Array) {
	return mlx.Zeros(mlx.DTypeFloat16, 1, 4, L, 8), mlx.Zeros(mlx.DTypeFloat16, 1, 4, L, 8)
}

// taggedKV builds an L-token K/V batch where every element of the token at
// absolute offset p carries the value p+1 (the +1 keeps tags distinct from the
// zero grow-padding the cache writes). The tag survives slicing/rotation, so a
// restored window's logical order can be read back as the sequence of absolute
// positions it holds. Shape is the standard [B=1, H=4, L, D=8].
func taggedKV(startOffset, L int) (*mlx.Array, *mlx.Array) {
	const H, D = 4, 8
	vals := make([]float32, H*L*D)
	for l := range L {
		tag := float32(startOffset + l + 1)
		for h := range H {
			for d := range D {
				vals[(h*L+l)*D+d] = tag
			}
		}
	}
	k := mlx.FromValues(vals, 1, H, L, D)
	v := mlx.FromValues(vals, 1, H, L, D)
	return k, v
}

// fillTagged advances c from offset 0 to n with single-token tagged writes (the
// decode update path), so each stored position carries its absolute-offset tag.
func fillTagged(c Attention, n int) {
	for p := range n {
		k, v := taggedKV(p, 1)
		c.Update(newKVBatch(p, 1), k, v)
	}
}

// windowTags reads c's logical window and returns the per-position tag (the
// absolute offset each slot holds, recovered as value-1). It uses element 0 of
// each token, which taggedKV set uniformly. Returns nil if the window is empty.
func windowTags(t *testing.T, c *RotatingKVCache) []int {
	t.Helper()
	state := c.State()
	if len(state) == 0 {
		return nil
	}
	k := state[0]
	K := k.Dim(2)
	if K == 0 {
		return nil
	}
	// Linearize ring storage into logical (oldest-first) order: slots
	// [oldest, K) ++ [0, oldest). After concat the buffer is already
	// in logical order (oldest == 0), so the concat below is skipped.
	if oldest := c.idx % K; oldest != 0 {
		tail := k.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(oldest, K), mlx.Slice())
		head := k.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, oldest), mlx.Slice())
		k = tail.Concatenate(2, head)
	}
	L, D := k.Dim(2), k.Dim(3)
	mlx.Eval(k)
	f := k.Floats()
	tags := make([]int, L)
	for l := range L {
		tags[l] = int(f[l*D]) - 1 // element 0 of token l; undo the +1 tag offset
	}
	return tags
}

// wantWindowTags returns the absolute positions a full window at offset would
// hold in logical (oldest-first) order: the trailing min(offset, window) tokens.
func wantWindowTags(offset, window int) []int {
	n := min(offset, window)
	tags := make([]int, n)
	for i := range tags {
		tags[i] = offset - n + i
	}
	return tags
}

// TestKVCachePerTokenSnapshotRestore schedules per-token offsets across a single
// batched write and verifies the captures are edge-local and that the
// speculation commit path — a live rewind, since KV is append-only — restores
// the cache to each accepted offset with the prefix intact.
func TestKVCachePerTokenSnapshotRestore(t *testing.T) {
	skipIfNoMLX(t)

	const before = 6
	const draft = 4

	for accepted := 0; accepted <= draft; accepted++ {
		c := NewKVCache()
		fillKV(c, before)

		offsets := make([]int, draft)
		for i := range offsets {
			offsets[i] = before + i
		}
		c.PrepareSnapshots(offsets)

		k, v := batchKV(draft)
		c.Update(newKVBatch(before, draft), k, v)
		if c.Offset() != before+draft {
			t.Fatalf("accepted=%d: offset after write = %d, want %d", accepted, c.Offset(), before+draft)
		}

		snaps := c.TakeSnapshots()
		if len(snaps) != draft {
			t.Fatalf("accepted=%d: got %d snapshots, want %d", accepted, len(snaps), draft)
		}

		// Captures are edge-local: offset before is zero-width (nil), and each
		// later offset holds exactly the single token [before+i-1, before+i).
		if snaps[0] != nil {
			t.Fatalf("accepted=%d: snaps[0] = %v, want nil (zero-width base)", accepted, snaps[0])
		}
		for i := 1; i < draft; i++ {
			ks := snaps[i].(*kvSnapshot)
			if ks.fromOffset != before+i-1 || ks.toOffset != before+i {
				t.Fatalf("accepted=%d: snaps[%d] = [%d,%d), want [%d,%d)", accepted, i, ks.fromOffset, ks.toOffset, before+i-1, before+i)
			}
		}

		// Commit rolls back via a live rewind (Restore(nil)) — the append-only
		// buffer still holds [0, before+draft), so the edge captures go unused.
		if accepted < draft {
			if !c.Restore(nil, before+accepted) {
				t.Fatalf("accepted=%d: live rewind failed", accepted)
			}
		}
		for _, s := range snaps {
			if s != nil {
				s.Close()
			}
		}

		want := before + draft
		if accepted < draft {
			want = before + accepted
		}
		if c.Offset() != want {
			t.Fatalf("accepted=%d: offset after commit = %d, want %d", accepted, c.Offset(), want)
		}
		if st := c.State(); len(st) == 2 && st[0].Dim(2) != want {
			t.Fatalf("accepted=%d: state seq dim = %d, want %d", accepted, st[0].Dim(2), want)
		}
	}
}

// TestKVCaptureMergeSplit verifies that two adjacent edge-local captures merge
// into the combined edge and split back into the halves — the operations the
// trie performs on stored snapshots (mergeWithChild / splitNode) — proving they
// work on captured snapshots, not just freshly-taken ones.
func TestKVCaptureMergeSplit(t *testing.T) {
	skipIfNoMLX(t)

	const before = 6
	c := NewKVCache()
	fillKV(c, before)

	// Schedule two interior offsets so the write captures [before, before+1) and
	// [before+1, before+2).
	c.PrepareSnapshots([]int{before + 1, before + 2})
	k, v := batchKV(3)
	c.Update(newKVBatch(before, 3), k, v)

	snaps := c.TakeSnapshots()
	a := snaps[0].(*kvSnapshot)
	b := snaps[1].(*kvSnapshot)
	if a.fromOffset != before || a.toOffset != before+1 {
		t.Fatalf("snaps[0] = [%d,%d), want [%d,%d)", a.fromOffset, a.toOffset, before, before+1)
	}
	if b.fromOffset != before+1 || b.toOffset != before+2 {
		t.Fatalf("snaps[1] = [%d,%d), want [%d,%d)", b.fromOffset, b.toOffset, before+1, before+2)
	}

	// Merge the adjacent edges into [before, before+2).
	merged := c.Merge(snaps[0], snaps[1]).(*kvSnapshot)
	if merged.fromOffset != before || merged.toOffset != before+2 {
		t.Fatalf("merged = [%d,%d), want [%d,%d)", merged.fromOffset, merged.toOffset, before, before+2)
	}

	// Split back at before+1 and confirm the halves match the originals.
	p, ch := c.Split(merged, before+1)
	ps := p.(*kvSnapshot)
	cs := ch.(*kvSnapshot)
	if ps.fromOffset != before || ps.toOffset != before+1 {
		t.Fatalf("split parent = [%d,%d), want [%d,%d)", ps.fromOffset, ps.toOffset, before, before+1)
	}
	if cs.fromOffset != before+1 || cs.toOffset != before+2 {
		t.Fatalf("split child = [%d,%d), want [%d,%d)", cs.fromOffset, cs.toOffset, before+1, before+2)
	}
	p.Close()
	ch.Close()
}

// TestRotatingPerTokenSnapshotRestore exercises per-token capture on a
// rotating cache across regimes: ring not yet full, exactly full, and wrapped.
// Every restore is exact-match against its own snapshot, so it must succeed
// regardless of wrap state, and the restored logical window must hold exactly
// the trailing absolute positions it should. Tagged K/V make slot-math errors
// observable as wrong positions, not just wrong shapes; the wrapped regime
// forces concat's linearize branch to run on entry.
func TestRotatingPerTokenSnapshotRestore(t *testing.T) {
	skipIfNoMLX(t)

	cases := []struct {
		name   string
		window int
		before int
	}{
		{"ring-not-full", 32, 4},
		{"ring-exactly-full", 8, 4},
		{"ring-wrapped", 4, 10},
	}

	const draft = 4
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for accepted := 0; accepted <= draft; accepted++ {
				c := NewRotatingKVCache(tc.window)
				fillTagged(c, tc.before)

				offsets := make([]int, draft)
				for i := range offsets {
					offsets[i] = tc.before + i
				}
				c.PrepareSnapshots(offsets)

				k, v := taggedKV(tc.before, draft)
				c.Update(newKVBatch(tc.before, draft), k, v)

				snaps := c.TakeSnapshots()
				if len(snaps) != draft {
					t.Fatalf("accepted=%d: got %d snapshots, want %d", accepted, len(snaps), draft)
				}

				if accepted < draft {
					if !c.Restore(snaps[accepted], tc.before+accepted) {
						t.Fatalf("accepted=%d: restore failed", accepted)
					}
				}
				for _, s := range snaps {
					s.Close()
				}

				want := tc.before + draft
				if accepted < draft {
					want = tc.before + accepted
				}
				if c.Offset() != want {
					t.Fatalf("accepted=%d: offset after commit = %d, want %d", accepted, c.Offset(), want)
				}
				// The logical window holds the trailing min(offset, window)
				// absolute positions in order — a slot-math error would keep the
				// right count but the wrong positions, which a dimension check
				// would miss. A batched write through concat can leave the raw
				// buffer larger than the window (it retains maxSize-1+L slots);
				// View trims to the trailing window at SDPA time, so compare the
				// trailing window of the linearized buffer.
				got := windowTags(t, c)
				if len(got) > tc.window {
					got = got[len(got)-tc.window:]
				}
				if wantTags := wantWindowTags(want, tc.window); !slices.Equal(got, wantTags) {
					t.Fatalf("accepted=%d: window tags = %v, want %v", accepted, got, wantTags)
				}
			}
		})
	}
}

// TestRotatingRestoreLazyOwnSnapshotSlices verifies the restore fast path:
// restoring from this cache's own still-lazy, non-trie-owned snapshot slices the
// live buffer rather than copying the window out, and does not consume the
// snapshot — it stays lazy, re-pointed at the new buffer, so it still names the
// same window data. Covers the restored content, a following decode write, and
// the snapshot copying out correctly afterward, across wrap regimes.
func TestRotatingRestoreLazyOwnSnapshotSlices(t *testing.T) {
	skipIfNoMLX(t)

	cases := []struct {
		name   string
		window int
		before int
	}{
		{"ring-not-full", 32, 4},
		{"ring-exactly-full", 8, 4},
		{"ring-wrapped", 4, 10},
	}

	const draft, accepted = 4, 2
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := NewRotatingKVCache(tc.window)
			fillTagged(c, tc.before)

			offsets := make([]int, draft)
			for i := range offsets {
				offsets[i] = tc.before + i
			}
			c.PrepareSnapshots(offsets)

			k, v := taggedKV(tc.before, draft)
			c.Update(newKVBatch(tc.before, draft), k, v)

			snaps := c.TakeSnapshots()
			snap := snaps[accepted].(*rotatingSnapshot)
			if snap.keys != nil {
				t.Fatal("snapshot materialized before restore; expected lazy")
			}

			if !c.Restore(snap, tc.before+accepted) {
				t.Fatal("restore failed")
			}

			// Fast path: the snapshot was sliced, not copied out, and stays a
			// valid lazy snapshot re-pointed at the new buffer.
			if snap.keys != nil {
				t.Fatal("snapshot was copied out; expected the slice fast path")
			}
			if snap.cache != c {
				t.Fatal("snapshot lost its lazy cache reference")
			}
			if !slices.Contains(c.lazySnapshots, snap) {
				t.Fatal("snapshot not re-added to the lazy set")
			}
			if snap.sliceStart != 0 || snap.sliceEnd != min(tc.before+accepted, tc.window) {
				t.Fatalf("snapshot slots = [%d,%d), want [0,%d)", snap.sliceStart, snap.sliceEnd, min(tc.before+accepted, tc.window))
			}

			got := windowTags(t, c)
			want := wantWindowTags(tc.before+accepted, tc.window)
			if !slices.Equal(got, want) {
				t.Fatalf("window tags = %v, want %v", got, want)
			}

			// A following decode write must produce correct content (the sliced
			// buffer feeds update's ring math unchanged). The write copies the
			// re-pointed snapshot out first; it must capture the same window.
			wk, wv := taggedKV(c.Offset(), 1)
			c.Update(newKVBatch(c.Offset(), 1), wk, wv)
			if snap.keys == nil {
				t.Fatal("following write did not copy out the re-pointed snapshot")
			}
			mlx.Eval(snap.keys)
			if head := int(snap.keys.Floats()[0]) - 1; head != tc.before+accepted-min(tc.before+accepted, tc.window) {
				t.Fatalf("re-pointed snapshot head tag = %d, want %d", head, tc.before+accepted-min(tc.before+accepted, tc.window))
			}
			got = windowTags(t, c)
			want = wantWindowTags(tc.before+accepted+1, tc.window)
			if !slices.Equal(got, want) {
				t.Fatalf("after follow-up write: window tags = %v, want %v", got, want)
			}

			for _, s := range snaps {
				if s != nil {
					s.Close()
				}
			}
		})
	}
}

// TestRotatingRestoreHookedSnapshotStaysLazy verifies a trie-owned snapshot (one
// with a materialize hook) takes the same re-point fast path: restore does not
// copy it out, so its hook does not fire and pagedOutBytes is not charged while
// the window still rides the live buffer. The hook fires exactly once, later,
// when a following write would destroy the window and copies the snapshot out —
// the lazy mechanism paying for itself only when the data is about to be lost.
func TestRotatingRestoreHookedSnapshotStaysLazy(t *testing.T) {
	skipIfNoMLX(t)

	const window, before, draft, accepted = 4, 10, 4, 2

	c := NewRotatingKVCache(window)
	fillTagged(c, before)

	offsets := make([]int, draft)
	for i := range offsets {
		offsets[i] = before + i
	}
	c.PrepareSnapshots(offsets)

	k, v := taggedKV(before, draft)
	c.Update(newKVBatch(before, draft), k, v)

	snaps := c.TakeSnapshots()
	snap := snaps[accepted].(*rotatingSnapshot)
	// Simulate trie ownership: a node sets a materialize hook on attach.
	fired := 0
	snap.SetMaterializeHook(func(int) { fired++ })

	if !c.Restore(snap, before+accepted) {
		t.Fatal("restore failed")
	}
	// Re-pointed, not copied out: still lazy, hook unfired.
	if snap.keys != nil {
		t.Fatal("hooked snapshot was copied out; expected the re-point fast path")
	}
	if fired != 0 {
		t.Fatalf("materialize hook fired %d times on restore, want 0 (still lazy)", fired)
	}

	got := windowTags(t, c)
	want := wantWindowTags(before+accepted, window)
	if !slices.Equal(got, want) {
		t.Fatalf("window tags = %v, want %v", got, want)
	}

	// A following decode write destroys the window's slots, so it copies the
	// snapshot out — firing the hook exactly once.
	wk, wv := taggedKV(c.Offset(), 1)
	c.Update(newKVBatch(c.Offset(), 1), wk, wv)
	if snap.keys == nil {
		t.Fatal("following write did not copy out the snapshot")
	}
	if fired != 1 {
		t.Fatalf("materialize hook fired %d times, want 1", fired)
	}

	for _, s := range snaps {
		if s != nil {
			s.Close()
		}
	}
}

// TestRotatingSnapshotSingleTokenWrite mirrors the tail of chunked prefill: the
// loop leaves the last token for decode seeding, so when two tokens remain it
// writes a single-token chunk (L=1) through update rather than concat. A snapshot
// scheduled at that write's end offset is captured against a buffer that may be in
// ring order, where a lazy slot slice would name the wrong slots. Covers the
// not-yet-wrapped and wrapped regimes; the wrapped one exercises the ring-clone
// fallback.
func TestRotatingSnapshotSingleTokenWrite(t *testing.T) {
	skipIfNoMLX(t)

	cases := []struct {
		name   string
		window int
		before int // tokens written before the final single-token write
	}{
		{"not-wrapped", 32, 5},
		{"wrapped", 4, 10},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := NewRotatingKVCache(tc.window)
			fillTagged(c, tc.before)

			// Schedule the offset the single-token write reaches as its end
			// boundary, then perform that write.
			c.PrepareSnapshots([]int{tc.before + 1})
			k, v := taggedKV(tc.before, 1)
			c.Update(newKVBatch(tc.before, 1), k, v)

			snaps := c.TakeSnapshots()
			if len(snaps) != 1 || snaps[0] == nil {
				t.Fatalf("snapshot not captured: %v", snaps)
			}

			if !c.Restore(snaps[0], tc.before+1) {
				t.Fatal("restore failed")
			}
			snaps[0].Close()

			if got, want := windowTags(t, c), wantWindowTags(tc.before+1, tc.window); !slices.Equal(got, want) {
				t.Fatalf("window tags = %v, want %v", got, want)
			}
		})
	}
}

// TestRotatingSnapshotSurvivesLaterChunk mirrors chunked prefill: a snapshot
// captured during one batched write must still restore correctly after a second
// batched write trims/rewrites the buffer the snapshot's slots lived in. This is
// the case that forces a lazy snapshot to copy out before the later write destroys it.
func TestRotatingSnapshotSurvivesLaterChunk(t *testing.T) {
	skipIfNoMLX(t)

	const window = 6
	c := NewRotatingKVCache(window)

	// Schedule an offset in the first chunk and one in the second, then write
	// both chunks before taking — the second chunk's concat trims past the first
	// snapshot's window.
	c.PrepareSnapshots([]int{4, 12})

	k1, v1 := taggedKV(0, 8) // chunk 1: [0, 8)
	c.Update(newKVBatch(0, 8), k1, v1)
	k2, v2 := taggedKV(8, 8) // chunk 2: [8, 16)
	c.Update(newKVBatch(8, 8), k2, v2)

	snaps := c.TakeSnapshots()
	if len(snaps) != 2 || snaps[0] == nil || snaps[1] == nil {
		t.Fatalf("snapshots not captured: %v", snaps)
	}

	// Restore the first-chunk snapshot (offset 4): its window predates the
	// second chunk entirely, so the data must have survived the chunk-2 write.
	if !c.Restore(snaps[0], 4) {
		t.Fatal("restore to offset 4 failed")
	}
	if got, want := windowTags(t, c), wantWindowTags(4, window); !slices.Equal(got, want) {
		t.Fatalf("offset 4 window tags = %v, want %v", got, want)
	}

	// Restore the second-chunk snapshot (offset 12) on a fresh cache.
	c2 := NewRotatingKVCache(window)
	if !c2.Restore(snaps[1], 12) {
		t.Fatal("restore to offset 12 failed")
	}
	if got, want := windowTags(t, c2), wantWindowTags(12, window); !slices.Equal(got, want) {
		t.Fatalf("offset 12 window tags = %v, want %v", got, want)
	}
	for _, s := range snaps {
		s.Close()
	}
}

// TestRotatingLazySnapshotSizeZeroUntilMaterialized verifies the speculation
// shape — a single batched write with per-token snapshots and no later write —
// leaves every interior capture lazy (no copy-out, so rejected drafts cost only
// the lazy arithmetic; the start boundary is the one eager clone), and that a
// lazy snapshot reports Size() == 0 until a destructive write copies it out, at
// which point the materialize hook fires with the newly-allocated bytes.
func TestRotatingLazySnapshotSizeZeroUntilMaterialized(t *testing.T) {
	skipIfNoMLX(t)

	const window = 4
	const before = 10 // wrapped
	const draft = 4
	c := NewRotatingKVCache(window)
	fillTagged(c, before)

	offsets := make([]int, draft)
	for i := range offsets {
		offsets[i] = before + i
	}
	c.PrepareSnapshots(offsets)

	k, v := taggedKV(before, draft)
	c.Update(newKVBatch(before, draft), k, v)

	snaps := c.TakeSnapshots()
	defer func() {
		for _, s := range snaps {
			if s != nil {
				s.Close()
			}
		}
	}()

	// Interior captures (offsets after the start boundary) stay lazy: no write
	// destroyed their slots, so keys is still nil and the issuing cache is live.
	for i := 1; i < draft; i++ {
		rs := snaps[i].(*rotatingSnapshot)
		if rs.keys != nil {
			t.Fatalf("snaps[%d] copied out (keys != nil); expected a live lazy snapshot", i)
		}
		if rs.cache == nil {
			t.Fatalf("snaps[%d] has no issuing cache; expected a live lazy snapshot", i)
		}
	}

	lazy := snaps[1].(*rotatingSnapshot)
	if lazy.Size() != 0 {
		t.Fatalf("lazy rotating snapshot Size = %d, want 0", lazy.Size())
	}

	var hookDelta int
	lazy.SetMaterializeHook(func(delta int) { hookDelta = delta })

	// Free the cache to force every outstanding lazy snapshot to copy out.
	c.Free()

	if lazy.keys == nil {
		t.Fatal("Free did not materialize the lazy snapshot")
	}
	want := lazy.keys.NumBytes() + lazy.values.NumBytes()
	if hookDelta != want {
		t.Fatalf("hook fired with delta %d, want %d", hookDelta, want)
	}
	if lazy.Size() != want {
		t.Fatalf("materialized Size = %d, want %d", lazy.Size(), want)
	}
}

// TestPerTokenSnapshotPersistsAcrossWrites verifies that scheduled offsets
// survive multiple writes until TakeSnapshots — the property prefill would rely
// on to snapshot interior offsets without splitting its forward.
func TestPerTokenSnapshotPersistsAcrossWrites(t *testing.T) {
	skipIfNoMLX(t)

	c := NewKVCache()
	fillKV(c, 2)

	// Schedule offsets that span two separate writes. Offset 2 equals the
	// schedule-time position, so it captures a zero-width range (nil); the rest
	// are edge-local.
	c.PrepareSnapshots([]int{2, 3, 5})

	k1, v1 := batchKV(2) // reaches offsets 2,3
	c.Update(newKVBatch(2, 2), k1, v1)
	k2, v2 := batchKV(2) // reaches offsets 4,5
	c.Update(newKVBatch(4, 2), k2, v2)

	snaps := c.TakeSnapshots()
	if len(snaps) != 3 {
		t.Fatalf("got %d snapshots, want 3", len(snaps))
	}
	if snaps[0] != nil {
		t.Fatalf("snaps[0] = %v, want nil (zero-width base)", snaps[0])
	}
	if s := snaps[1].(*kvSnapshot); s.fromOffset != 2 || s.toOffset != 3 {
		t.Fatalf("snaps[1] = [%d,%d), want [2,3)", s.fromOffset, s.toOffset)
	}
	// Offset 5 was scheduled across the second write; its edge starts at the
	// previous scheduled offset (3), confirming the base cursor only advances
	// on capture so the snapshot range matches the trie edge between scheduled
	// offsets — write boundaries between captures must not move it.
	if s := snaps[2].(*kvSnapshot); s.fromOffset != 3 || s.toOffset != 5 {
		t.Fatalf("snaps[2] = [%d,%d), want [3,5)", s.fromOffset, s.toOffset)
	}

	// Restore from the [2,3) edge snapshot to offset 3.
	if !c.Restore(snaps[1], 3) {
		t.Fatal("restore to offset 3 failed")
	}
	if c.Offset() != 3 {
		t.Fatalf("offset = %d, want 3", c.Offset())
	}
	for _, s := range snaps {
		if s != nil {
			s.Close()
		}
	}
}

// TestRecurrentSnapshotSplitsAndSegmentedCapture verifies that SnapshotSplits
// reports the interior scheduled offsets and PutSegmented captures them from the
// per-boundary states, so each accepted count restores to a distinct state.
func TestRecurrentSnapshotSplitsAndSegmentedCapture(t *testing.T) {
	skipIfNoMLX(t)

	const convTail, convDim, nv, vd, kd = 3, 8, 2, 4, 4
	c := NewRecurrentCache(convTail, convDim, nv, vd, kd)
	c.Get(newKVBatch(0, 1), mlx.DTypeFloat16)
	// Advance to offset 5 so the speculative forward starts there.
	c.Put(newKVBatch(0, 5),
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat16, 1, convTail, convDim)},
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat32, 1, nv, vd, kd)})

	const before, draft = 5, 4
	offsets := []int{before, before + 1, before + 2, before + 3}
	c.PrepareSnapshots(offsets)

	splits := c.SnapshotSplits(draft)
	want := []int{1, 2, 3}
	if len(splits) != len(want) {
		t.Fatalf("SnapshotSplits = %v, want %v", splits, want)
	}
	for i := range want {
		if splits[i] != want[i] {
			t.Fatalf("SnapshotSplits = %v, want %v", splits, want)
		}
	}

	// Distinct per-boundary states (3 interior splits + the end) so restore
	// targets are distinguishable.
	mkConv := func(s float32) *mlx.Array { return mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat16, 1, convTail, convDim), s) }
	mkDelta := func(s float32) *mlx.Array { return mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, nv, vd, kd), s) }
	convStates := []*mlx.Array{mkConv(1), mkConv(2), mkConv(3), mkConv(4)}
	deltaStates := []*mlx.Array{mkDelta(1), mkDelta(2), mkDelta(3), mkDelta(4)}

	c.Put(newKVBatch(before, draft), convStates, deltaStates)
	if c.Offset() != before+draft {
		t.Fatalf("offset after segmented put = %d, want %d", c.Offset(), before+draft)
	}

	snaps := c.TakeSnapshots()
	if len(snaps) != draft {
		t.Fatalf("got %d snapshots, want %d", len(snaps), draft)
	}
	for i, s := range snaps {
		if s == nil {
			t.Fatalf("snapshot %d not captured", i)
		}
	}

	// Full accept (no restore): the live state is the committed end boundary
	// (value 4) at offset before+draft.
	st := c.State()
	mlx.Eval(st[1])
	if got := st[1].Floats()[0]; got != 4 {
		t.Fatalf("full-accept delta state = %v, want end boundary value 4", got)
	}

	// Each partial accept restores to offset before+accepted and must recover the
	// distinct boundary state captured there: snaps[0] is the pre-forward state
	// (value 0); snaps[i>=1] is the interior split boundary (value i). Recurrent
	// snapshots are self-contained, so restores need not run in order.
	for accepted := range draft {
		if !c.Restore(snaps[accepted], before+accepted) {
			t.Fatalf("accepted=%d: restore to before+%d failed", accepted, accepted)
		}
		if c.Offset() != before+accepted {
			t.Fatalf("accepted=%d: offset after restore = %d, want %d", accepted, c.Offset(), before+accepted)
		}
		st := c.State()
		mlx.Eval(st[1])
		if got := st[1].Floats()[0]; got != float32(accepted) {
			t.Fatalf("accepted=%d: restored delta state = %v, want boundary value %d", accepted, got, accepted)
		}
	}
	for _, s := range snaps {
		s.Close()
	}
}

// TestPrepareSnapshotsPastOffsetPanics verifies scheduling an already-passed offset
// is rejected.
func TestPrepareSnapshotsPastOffsetPanics(t *testing.T) {
	skipIfNoMLX(t)

	c := NewKVCache()
	fillKV(c, 5)

	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for already-passed offset")
		}
	}()
	c.PrepareSnapshots([]int{3})
}
