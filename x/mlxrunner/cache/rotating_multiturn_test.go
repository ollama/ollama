package cache

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// singleTokenKV and multiTokenKV fabricate [B=1, H=1, L, D=2] key/value
// tensors whose channel value is the token id, so stateIDs can recover
// which ids survived in the cache.
func singleTokenKV(id float32) (*mlx.Array, *mlx.Array) {
	k := mlx.FromValues([]float32{id, id}, 1, 1, 1, 2)
	v := mlx.FromValues([]float32{id, id}, 1, 1, 1, 2)
	return k, v
}

func multiTokenKV(ids []float32) (*mlx.Array, *mlx.Array) {
	data := make([]float32, 0, 2*len(ids))
	for _, id := range ids {
		data = append(data, id, id)
	}
	k := mlx.FromValues(data, 1, 1, len(ids), 2)
	v := mlx.FromValues(data, 1, 1, len(ids), 2)
	return k, v
}

// stateIDs returns the ids currently in the cache in slot order (logical
// after a concat, physical/rotated after a single-token update).
func stateIDs(t *testing.T, c *RotatingKVCache) []float32 {
	t.Helper()
	state := c.State()
	if state == nil {
		return nil
	}
	mlx.Eval(state[0])
	flat := state[0].Floats()
	n := state[0].Dim(2)
	out := make([]float32, n)
	for i := range n {
		out[i] = flat[i*2]
	}
	return out
}

func equalSlice(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func feedMulti(c *RotatingKVCache, startID float32, n int) float32 {
	ids := make([]float32, n)
	for i := range ids {
		ids[i] = startID + float32(i)
	}
	k, v := multiTokenKV(ids)
	c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	return startID + float32(n)
}

func feedSingle(c *RotatingKVCache, id float32) {
	k, v := singleTokenKV(id)
	c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
}

// TestRotatingKVCacheConcatMidRotationPreservesContext: after the buffer
// has wrapped, a multi-token concat must keep the (maxSize-1) most recent
// pre-existing tokens in logical order so the first Q of the new batch
// has a full sliding window.
func TestRotatingKVCacheConcatMidRotationPreservesContext(t *testing.T) {
	skipIfNoMLX(t)

	const window = 4
	c := NewRotatingKVCache(window)

	nextID := feedMulti(c, 1, 3)
	for range 6 {
		feedSingle(c, nextID)
		nextID++
	}
	if c.Offset() != 9 {
		t.Fatalf("setup: offset=%d want 9", c.Offset())
	}
	if c.idx >= c.maxSize {
		t.Fatalf("setup: expected mid-rotation idx (<%d), got %d", c.maxSize, c.idx)
	}

	feedMulti(c, 10, 2)
	got := stateIDs(t, c)
	want := []float32{7, 8, 9, 10, 11}
	if !equalSlice(got, want) {
		t.Fatalf("post-concat window=%v want %v", got, want)
	}
	if c.Offset() != 11 {
		t.Fatalf("offset=%d want 11", c.Offset())
	}
}

// TestRotatingKVCacheConcatAlignedInvariant: with an aligned buffer
// (c.idx == Dim), an L>1 concat keeps the last (maxSize-1) pre-existing
// tokens plus the full new batch. This is the chunked-prefill contract
// x/mlxrunner/pipeline.go relies on.
func TestRotatingKVCacheConcatAlignedInvariant(t *testing.T) {
	skipIfNoMLX(t)

	const window = 4
	c := NewRotatingKVCache(window)

	// Chunk 1 fills past maxSize, leaving Dim == maxSize aligned.
	feedMulti(c, 1, 6)
	// Chunk 2: the buffer is intentionally oversized to (maxSize-1) + L
	// so the first new Q has its full window in scope for this forward.
	feedMulti(c, 7, 3)
	got := stateIDs(t, c)
	want := []float32{4, 5, 6, 7, 8, 9}
	if !equalSlice(got, want) {
		t.Fatalf("post-chunk-2 buffer=%v want %v", got, want)
	}

	// The next decode trims oversize back to maxSize; order may be
	// physical (rotated), so check as a set.
	feedSingle(c, 10)
	got = stateIDs(t, c)
	if len(got) != window {
		t.Fatalf("post-decode Dim=%d want %d", len(got), window)
	}
	seen := map[float32]bool{}
	for _, v := range got {
		seen[v] = true
	}
	for _, w := range []float32{7, 8, 9, 10} {
		if !seen[w] {
			t.Fatalf("post-decode window missing %v (got %v)", w, got)
		}
	}
}

// TestRotatingKVCacheConcatAfterDecodeGrowsBuffer: update() grows the
// underlying buffer by `step` slots via mlx.Zeros before writing, so
// after one decode on a short prefill c.idx < Dim even though the cache
// has not wrapped. Those trailing slots are zero padding and must not
// be pulled back into the live window on the next concat.
func TestRotatingKVCacheConcatAfterDecodeGrowsBuffer(t *testing.T) {
	skipIfNoMLX(t)

	const window = 512
	c := NewRotatingKVCache(window)

	feedMulti(c, 1, 3)
	feedSingle(c, 4)
	feedMulti(c, 5, 3)

	got := stateIDs(t, c)
	want := []float32{1, 2, 3, 4, 5, 6, 7}
	if !equalSlice(got, want) {
		t.Fatalf("growing-buffer concat=%v want %v", got, want)
	}
}

// TestRotatingKVCacheConcatAfterLiveRewind: x/mlxrunner/cache.go calls
// Restore(nil, target) between conversation turns to rewind the cache to
// the matched prefix. Restore moves c.offset/c.idx without trimming the
// underlying buffer, so slots [c.idx, Dim) still hold stale pre-rewind
// tokens. A subsequent concat must drop those, not treat them as wrapped
// window content.
func TestRotatingKVCacheConcatAfterLiveRewind(t *testing.T) {
	skipIfNoMLX(t)

	const window = 8
	c := NewRotatingKVCache(window)

	// Grow the buffer to exactly maxSize without wrapping.
	feedMulti(c, 1, 2)
	for id := float32(3); id <= 8; id++ {
		feedSingle(c, id)
	}
	if c.Offset() != window {
		t.Fatalf("setup: offset=%d want %d", c.Offset(), window)
	}

	if !c.Restore(nil, 2) {
		t.Fatalf("live rewind to 2 failed")
	}
	if c.Offset() != 2 {
		t.Fatalf("post-rewind offset=%d want 2", c.Offset())
	}

	feedMulti(c, 9, 3)
	got := stateIDs(t, c)
	want := []float32{1, 2, 9, 10, 11}
	if !equalSlice(got, want) {
		t.Fatalf("post-rewind concat=%v want %v", got, want)
	}
	if c.Offset() != 5 {
		t.Fatalf("offset=%d want 5", c.Offset())
	}
}

// TestRotatingKVCacheConcatGrowingBuffer: when oldLen < maxSize the trim
// formula drops to non-positive and all pre-existing tokens are kept.
func TestRotatingKVCacheConcatGrowingBuffer(t *testing.T) {
	skipIfNoMLX(t)

	const window = 4
	c := NewRotatingKVCache(window)

	feedMulti(c, 1, 2)
	feedMulti(c, 3, 2)
	got := stateIDs(t, c)
	want := []float32{1, 2, 3, 4}
	if !equalSlice(got, want) {
		t.Fatalf("growing buffer=%v want %v", got, want)
	}
}

// TestRotatingKVCacheRunnerChunkedPrefill mirrors the
// x/mlxrunner/pipeline.go prefill loop: a long prompt fed through
// repeated L>1 Update() calls on a single cache. Scaled-down proxy for
// the Gemma 4 26B case (sliding_window=1024, prefillChunkSize=2048).
func TestRotatingKVCacheRunnerChunkedPrefill(t *testing.T) {
	skipIfNoMLX(t)

	const window = 4
	c := NewRotatingKVCache(window)

	feedMulti(c, 1, 8)
	if c.Offset() != 8 {
		t.Fatalf("chunk 1: offset=%d want 8", c.Offset())
	}

	feedMulti(c, 9, 8)
	got := stateIDs(t, c)
	want := []float32{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	if !equalSlice(got, want) {
		t.Fatalf("chunk 2: buffer=%v want %v", got, want)
	}

	feedMulti(c, 17, 4)
	got = stateIDs(t, c)
	want = []float32{14, 15, 16, 17, 18, 19, 20}
	if !equalSlice(got, want) {
		t.Fatalf("chunk 3: buffer=%v want %v", got, want)
	}

	// Decode trims oversize back to maxSize; order may be physical.
	feedSingle(c, 21)
	got = stateIDs(t, c)
	if len(got) != window {
		t.Fatalf("post-decode Dim=%d want %d", len(got), window)
	}
	seen := map[float32]bool{}
	for _, v := range got {
		seen[v] = true
	}
	for _, w := range []float32{18, 19, 20, 21} {
		if !seen[w] {
			t.Fatalf("post-decode window missing %v (got %v)", w, got)
		}
	}
}

// TestRotatingKVCacheMultiTurnChatSimulation walks a prefill → decode →
// prefill sequence and checks that each new prefill retains the last
// (maxSize-1) pre-existing tokens in logical order.
func TestRotatingKVCacheMultiTurnChatSimulation(t *testing.T) {
	skipIfNoMLX(t)

	const window = 4
	c := NewRotatingKVCache(window)

	nextID := feedMulti(c, 1, 2)
	for range 5 {
		feedSingle(c, nextID)
		nextID++
	}
	if c.Offset() != 7 {
		t.Fatalf("turn 1: offset=%d want 7", c.Offset())
	}

	feedMulti(c, nextID, 3)
	nextID += 3
	got := stateIDs(t, c)
	want := []float32{5, 6, 7, 8, 9, 10}
	if !equalSlice(got, want) {
		t.Fatalf("turn 2 prefill buffer=%v want %v", got, want)
	}

	for range 4 {
		feedSingle(c, nextID)
		nextID++
	}
	if c.Offset() != 14 {
		t.Fatalf("turn 2 decode: offset=%d want 14", c.Offset())
	}

	feedMulti(c, nextID, 2)
	got = stateIDs(t, c)
	want = []float32{12, 13, 14, 15, 16}
	if !equalSlice(got, want) {
		t.Fatalf("turn 3 prefill buffer=%v want %v", got, want)
	}
}

// TestRotatingKVCacheOffsetTracking: Offset() is the monotonic logical
// token count through any mix of Update() calls — Gemma 4 uses
// donorEntry.Offset - L for the consumer's RoPE offset.
func TestRotatingKVCacheOffsetTracking(t *testing.T) {
	skipIfNoMLX(t)

	c := NewRotatingKVCache(4)
	nextID := feedMulti(c, 1, 3)
	if c.Offset() != 3 {
		t.Fatalf("after prefill 3: offset=%d want 3", c.Offset())
	}
	for i := range 5 {
		feedSingle(c, nextID)
		nextID++
		if c.Offset() != 3+i+1 {
			t.Fatalf("after decode %d: offset=%d want %d", i, c.Offset(), 3+i+1)
		}
	}
	nextID = feedMulti(c, nextID, 2)
	if c.Offset() != 10 {
		t.Fatalf("after turn-2 prefill: offset=%d want 10", c.Offset())
	}
	// L > maxSize concat.
	feedMulti(c, nextID, 7)
	if c.Offset() != 17 {
		t.Fatalf("after large prefill: offset=%d want 17", c.Offset())
	}
}
