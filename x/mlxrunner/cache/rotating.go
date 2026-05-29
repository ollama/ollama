package cache

import (
	"fmt"
	"slices"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// RotatingKVCache implements sliding window attention with bounded memory.
type RotatingKVCache struct {
	keys, values *mlx.Array
	offset       int
	step         int
	maxSize      int
	idx          int

	snapshots pendingSnapshots

	// lazySnapshots are outstanding snapshots still in their lazy state: they
	// index into the live keys/values buffer by slot rather than owning a copy
	// (see rotatingSnapshot). A write that trims, linearizes, or overwrites the
	// buffer copies them out (copyOutLazySnapshots) before destroying the slots
	// they name.
	lazySnapshots []*rotatingSnapshot
}

func NewRotatingKVCache(maxSize int) *RotatingKVCache {
	return &RotatingKVCache{maxSize: maxSize, step: 256}
}

// Assumes B = 1; heterogeneous batches are not supported.
func (c *RotatingKVCache) Update(b *batch.Batch, keys, values *mlx.Array) *nn.KVHistory {
	start := c.offset
	c.captureStartBoundary(start)

	batched := keys.Dim(2) > 1
	var newK, newV *mlx.Array
	if batched {
		newK, newV = c.concat(keys, values)
	} else {
		newK, newV = c.update(keys, values)
	}

	c.captureLazySnapshots(start, c.offset, batched)
	return nn.NewKVHistory(newK, newV, rotatingApplier{
		b:       b,
		K:       newK.Dim(2),
		L:       keys.Dim(2),
		window:  c.maxSize,
		ringIdx: c.idx,
		dtype:   keys.DType(),
	})
}

func (c *RotatingKVCache) concat(keys, values *mlx.Array) (newK *mlx.Array, newV *mlx.Array) {
	logutil.Trace("(*RotatingKVCache).concat", "keys_dim", keys.Dims(), "values_dim", values.Dims(), "offset", c.offset, "idx", c.idx, "max_size", c.maxSize)

	// Freeze outstanding lazy snapshots: the linearize/trim/concat below
	// reorders and drops the slots they name.
	c.copyOutLazySnapshots()

	if c.keys == nil {
		c.keys, c.values = keys.Clone(), values.Clone()
		mlx.Pin(c.keys, c.values)
	} else {
		if c.idx < c.keys.Dim(2) {
			if c.offset <= c.maxSize {
				// Not yet wrapped: slots [c.idx, Dim) are grow padding
				// or stale post-rewind data, not live window content.
				c.keys.Set(c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.idx), mlx.Slice()))
				c.values.Set(c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.idx), mlx.Slice()))
			} else {
				// Wrapped: logical order is slots[idx..Dim) then slots[0..idx).
				// Linearize so the trim + concat below operate on contiguous
				// positions and preserve the last (maxSize - 1) old tokens.
				tailK := c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(c.idx, c.keys.Dim(2)), mlx.Slice())
				tailV := c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(c.idx, c.values.Dim(2)), mlx.Slice())
				headK := c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.idx), mlx.Slice())
				headV := c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.idx), mlx.Slice())
				c.keys.Set(tailK.Concatenate(2, headK))
				c.values.Set(tailV.Concatenate(2, headV))
				c.idx = c.keys.Dim(2)
			}
		}

		// Trim to max_size to maintain sliding window
		if trim := c.idx - c.maxSize + 1; trim > 0 {
			c.keys.Set(c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(trim, c.keys.Dim(2)), mlx.Slice()))
			c.values.Set(c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(trim, c.values.Dim(2)), mlx.Slice()))
		}

		c.keys.Set(c.keys.Concatenate(2, keys))
		c.values.Set(c.values.Concatenate(2, values))
	}

	c.offset += keys.Dim(2)
	c.idx = c.keys.Dim(2)
	return c.keys, c.values
}

func (c *RotatingKVCache) update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	logutil.Trace("(*RotatingKVCache).update", "keys_dim", keys.Dims(), "values_dim", values.Dims(), "offset", c.offset, "idx", c.idx, "max_size", c.maxSize)

	// Freeze outstanding lazy snapshots: the trim/rotate/SliceUpdate below
	// overwrites the slots they name.
	c.copyOutLazySnapshots()

	B, H, L, Dk, Dv := keys.Dim(0), keys.Dim(1), keys.Dim(2), keys.Dim(3), values.Dim(3)

	prev := c.offset

	// Grow buffer if not yet at max
	if c.keys == nil || (prev >= c.keys.Dim(2) && c.keys.Dim(2) < c.maxSize) {
		newSize := min(c.step, c.maxSize-prev)
		newKeys := mlx.Zeros(keys.DType(), B, H, newSize, Dk)
		newValues := mlx.Zeros(values.DType(), B, H, newSize, Dv)
		if c.keys != nil {
			c.keys.Set(c.keys.Concatenate(2, newKeys))
			c.values.Set(c.values.Concatenate(2, newValues))
		} else {
			c.keys, c.values = newKeys, newValues
			mlx.Pin(c.keys, c.values)
		}
		c.idx = prev
	}

	// Trim to max_size to maintain sliding window
	if trim := c.keys.Dim(2) - c.maxSize; trim > 0 {
		c.keys.Set(c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(trim, c.keys.Dim(2)), mlx.Slice()))
		c.values.Set(c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(trim, c.values.Dim(2)), mlx.Slice()))
		c.idx = c.maxSize
	}

	// Rotate when hitting max
	if c.idx >= c.maxSize {
		c.idx = 0
	}

	c.keys.Set(c.keys.SliceUpdate(keys, mlx.Slice(), mlx.Slice(), mlx.Slice(c.idx, c.idx+L), mlx.Slice()))
	c.values.Set(c.values.SliceUpdate(values, mlx.Slice(), mlx.Slice(), mlx.Slice(c.idx, c.idx+L), mlx.Slice()))

	c.offset += L
	c.idx += L

	validLen := min(c.offset, c.maxSize)
	return c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, validLen), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, validLen), mlx.Slice())
}

// View returns the current cache contents as a read-only KV history, used by an
// assistant model that shares this cache. It sets L=1 so rotatingApplier treats
// the buffer as ring-ordered (its stored layout); L=1 is a layout selector, not
// a query length. A post-concat oversize buffer (K > maxSize) is already in
// logical order, so View trims to the trailing maxSize tokens and resets ringIdx
// to 0, collapsing the applier's gather to identity.
func (c *RotatingKVCache) View(b *batch.Batch) *nn.KVHistory {
	state := c.State()
	k, v := state[0], state[1]
	K := k.Dim(2)
	ringIdx := c.idx
	if K > c.maxSize {
		// Post-concat oversize buffer: storage is in logical (oldest-first)
		// order, so slice the trailing maxSize tokens. The slice is already
		// in logical layout, so reset ringIdx to make the applier's gather
		// collapse to identity.
		start := K - c.maxSize
		k = k.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, K), mlx.Slice())
		v = v.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, K), mlx.Slice())
		K = c.maxSize
		ringIdx = 0
	}
	return nn.NewKVHistory(k, v, rotatingApplier{
		b:       b,
		K:       K,
		L:       1,
		window:  c.maxSize,
		ringIdx: ringIdx,
		dtype:   k.DType(),
	})
}

func (c *RotatingKVCache) State() []*mlx.Array {
	if c.keys == nil || c.values == nil {
		return nil
	}
	liveLen := min(c.offset, c.keys.Dim(2))
	return []*mlx.Array{
		c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, liveLen), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, liveLen), mlx.Slice()),
	}
}

// replaceBuffer swaps in newK/newV as the cache's keys/values, unpinning the old
// buffer and pinning the new one.
func (c *RotatingKVCache) replaceBuffer(newK, newV *mlx.Array) {
	mlx.Unpin(c.keys, c.values)
	c.keys, c.values = newK, newV
	mlx.Pin(c.keys, c.values)
}

func (c *RotatingKVCache) Free() {
	// Freeing drops the buffer lazy snapshots index into; copy them out first.
	c.copyOutLazySnapshots()
	mlx.Unpin(c.keys, c.values)
	c.keys, c.values = nil, nil
	c.offset = 0
	c.idx = 0
	c.snapshots = pendingSnapshots{}
}

func (c *RotatingKVCache) Offset() int { return c.offset }

func (c *RotatingKVCache) PrepareSnapshots(offsets []int) { c.snapshots.prepare(c.offset, offsets) }
func (c *RotatingKVCache) TakeSnapshots() []Snapshot      { return c.snapshots.take() }

// captureStartBoundary captures a scheduled offset at the pre-write position via
// the clone path (Snapshot), so the rollback point holds the full pre-write
// window before concat/update reorders or drops it.
func (c *RotatingKVCache) captureStartBoundary(start int) {
	if len(c.snapshots.offsets) == 0 {
		return
	}
	c.snapshots.captureReached(start, func(int) Snapshot { return c.Snapshot(0) })
}

// captureLazySnapshots records a snapshot for each scheduled offset the write
// reached after the start boundary. A batched write (concat) linearizes the
// buffer into logical order, so each window is a contiguous slot slice captured
// lazily (lazyRotatingSnapshot). A single-token write (update) leaves the buffer
// ring-ordered or grown, breaking that slot math, so capture a clone (Snapshot)
// instead; update only ever reaches the end boundary (o == c.offset).
func (c *RotatingKVCache) captureLazySnapshots(start, end int, batched bool) {
	if len(c.snapshots.offsets) == 0 {
		return
	}
	for _, o := range c.snapshots.scheduledIn(start, end) {
		if o == start {
			continue // captured pre-write by captureStartBoundary
		}
		c.snapshots.captureReached(o, func(int) Snapshot {
			if batched {
				return c.lazyRotatingSnapshot(o)
			}
			return c.Snapshot(o - min(o, c.maxSize))
		})
	}
}

// lazyRotatingSnapshot records the window ending at offset o as a lazy snapshot
// into the current (logically ordered) buffer: window [o-liveLen, o) maps to
// slots [sliceStart, sliceEnd), and restoring sets idx == liveLen so the buffer
// reads back in logical order. Returns nil for a zero-width range.
func (c *RotatingKVCache) lazyRotatingSnapshot(o int) Snapshot {
	if c.keys == nil {
		return nil
	}
	bufBase := c.offset - c.keys.Dim(2)
	liveLen := min(o, c.maxSize)
	sliceStart := o - liveLen - bufBase
	sliceEnd := o - bufBase
	if sliceEnd <= sliceStart {
		return nil
	}
	s := &rotatingSnapshot{
		fromOffset: o - liveLen,
		toOffset:   o,
		idx:        liveLen,
		cache:      c,
		sliceStart: sliceStart,
		sliceEnd:   sliceEnd,
	}
	c.lazySnapshots = append(c.lazySnapshots, s)
	return s
}

// rotatingSnapshot holds paged-out data for a RotatingKVCache. Initially lazy:
// the window lives in the issuing cache's buffer at slots [sliceStart, sliceEnd)
// in logical order, and copyOut clones it into owned keys/values before a write
// reorders or drops those slots.
type rotatingSnapshot struct {
	keys, values         *mlx.Array // owned window once copied out; nil while lazy
	fromOffset, toOffset int        // absolute offset range the window covers
	idx                  int        // buffer write position a restore installs

	cache                *RotatingKVCache // issuer while lazy; nil once copied out
	sliceStart, sliceEnd int              // buffer slot range of the window while lazy

	// onMaterialize, if set, is fired once from copyOut with the newly-owned
	// byte count so an owner (e.g. the trie's pagedOutBytes counter) can pick
	// up bytes that were free while the snapshot was lazy.
	onMaterialize func(delta int)
}

func (s *rotatingSnapshot) Size() int {
	if s.keys != nil {
		return s.keys.NumBytes() + s.values.NumBytes()
	}
	// Lazy snapshots own no extra memory: the window still lives in the
	// issuing cache's buffer.
	return 0
}

func (s *rotatingSnapshot) SetMaterializeHook(fn func(delta int)) { s.onMaterialize = fn }

func (s *rotatingSnapshot) Close() {
	mlx.Unpin(s.keys, s.values)
	if s.cache != nil {
		s.cache.dropLazySnapshot(s)
		s.cache = nil
	}
}

// copyOut converts a lazy snapshot into an owned clone of its window slots. It
// is a no-op once the snapshot already owns its data. The slots hold the window
// in logical order, so the clone needs no reordering.
func (s *rotatingSnapshot) copyOut() {
	if s.keys != nil {
		return
	}
	c := s.cache
	kSlice := c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(s.sliceStart, s.sliceEnd), mlx.Slice())
	vSlice := c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(s.sliceStart, s.sliceEnd), mlx.Slice())
	k := mlx.Contiguous(kSlice, false)
	v := mlx.Contiguous(vSlice, false)
	mlx.Pin(k, v)
	mlx.AsyncEval(k, v)

	s.keys, s.values = k, v
	c.dropLazySnapshot(s)
	s.cache = nil

	if s.onMaterialize != nil {
		s.onMaterialize(s.keys.NumBytes() + s.values.NumBytes())
		s.onMaterialize = nil
	}
}

func (c *RotatingKVCache) dropLazySnapshot(s *rotatingSnapshot) {
	if i := slices.Index(c.lazySnapshots, s); i >= 0 {
		c.lazySnapshots = slices.Delete(c.lazySnapshots, i, i+1)
	}
}

// copyOutLazySnapshots clones every outstanding lazy snapshot into owned data.
// The destructive write paths (concat, update) call this before they trim,
// linearize, or overwrite the slots a snapshot names. copyOut removes the
// snapshot from the set, so iterate over a clone.
func (c *RotatingKVCache) copyOutLazySnapshots() {
	for _, s := range slices.Clone(c.lazySnapshots) {
		s.copyOut()
	}
}

func (c *RotatingKVCache) Snapshot(fromOffset int) Snapshot {
	if c.keys == nil || c.offset <= fromOffset {
		return nil
	}

	state := c.State()
	k := state[0].Clone()
	v := state[1].Clone()
	mlx.Pin(k, v)

	return &rotatingSnapshot{
		keys:       k,
		values:     v,
		fromOffset: fromOffset,
		toOffset:   c.offset,
		idx:        c.idx,
	}
}

func (c *RotatingKVCache) Restore(snapshot Snapshot, target int) bool {
	if target < 0 {
		return false
	}

	if snapshot == nil {
		if target >= c.offset {
			return target == c.offset
		}
		// Live rewind is only safe before the buffer fills (offset <= maxSize);
		// once wrapped, rewinding leaves an incomplete window, so a snapshot is
		// required.
		if c.offset > c.maxSize {
			return false
		}
		c.offset = target
		c.idx = target
		return true
	}

	snap := snapshot.(*rotatingSnapshot)

	if target > snap.toOffset {
		return false
	}

	// Reject if clamping would leave an incomplete window.
	if target < snap.toOffset && snap.toOffset > c.maxSize {
		return false
	}

	// Fast path: this cache's own still-lazy snapshot names the restored window
	// in the live buffer, so slice it in as the new buffer (no copy) and re-point
	// the snapshot to slots [0, liveLen), kept lazy.
	if snap.cache == c && snap.keys == nil {
		// Drop snap (re-pointed, not copied), then copy out the siblings: their
		// slots fall outside the restored window.
		c.dropLazySnapshot(snap)
		c.copyOutLazySnapshots()
		liveLen := snap.sliceEnd - snap.sliceStart
		c.replaceBuffer(
			c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(snap.sliceStart, snap.sliceEnd), mlx.Slice()),
			c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(snap.sliceStart, snap.sliceEnd), mlx.Slice()),
		)
		snap.sliceStart, snap.sliceEnd = 0, liveLen
		c.lazySnapshots = append(c.lazySnapshots, snap)
		c.offset = snap.toOffset
		c.idx = snap.idx
		if target < c.offset {
			c.offset = target
			c.idx = target
		}
		return true
	}

	// snap is owned (cross-cache or already materialized; the lazy-own case is
	// above), so copyOut is a no-op here. Freeze this cache's lazy snapshots off
	// the buffer the assignment below replaces.
	snap.copyOut()
	c.copyOutLazySnapshots()

	c.replaceBuffer(snap.keys.Clone(), snap.values.Clone())
	c.offset = snap.toOffset
	c.idx = snap.idx

	// Clamp to target if needed.
	if target < c.offset {
		c.offset = target
		c.idx = target
	}
	return true
}

func (c *RotatingKVCache) Merge(parent, child Snapshot) Snapshot {
	// For rotating caches, the child snapshot supersedes the parent
	// since it contains the full window state.
	if parent != nil {
		parent.Close()
	}
	return child
}

func (c *RotatingKVCache) Split(snapshot Snapshot, at int) (Snapshot, Snapshot) {
	// Rotating cache snapshots contain the full window state.
	// Cannot cleanly split a ring buffer at an arbitrary point.
	return nil, snapshot
}

// rotatingApplier composes the sliding-window storage restriction onto the
// caller's logical mask. ringIdx is the write cursor at Update time: at L=1
// decode the ring is not position-ordered (logical col j lives at slot
// (ringIdx+j) mod K), so tensor masks must be gathered into ring layout. At L>1
// prefill concat has linearized storage, so the gather is identity.
type rotatingApplier struct {
	b       *batch.Batch
	K       int
	L       int
	window  int
	ringIdx int
	dtype   mlx.DType
}

func (r rotatingApplier) ApplyMask(logical nn.AttentionMask) nn.AttentionMask {
	if r.L == 1 {
		// Single-query decode: storage already enforces the window and every
		// stored key's position <= absQ, so a zero or causal logical mask
		// reduces to no mask — let SDPA dispatch to mode="".
		if logical.IsZero() || logical.IsCausal() {
			return nn.AttentionMask{}
		}

		// Tensor-backed mask: materialize in logical order, then gather K cols
		// into ring-slot order to align with the cache output.
		arr := logical.AsArray(r.b, r.K, r.dtype)
		arr = gatherRingCols(arr, r.ringIdx, r.K)
		return nn.ArrayMask(arr)
	}

	return logical.Intersect(nn.SlidingWindowMask(r.b, r.K, r.window, r.dtype))
}

// gatherRingCols reorders a [B, 1, L, K] mask's K axis from logical order
// (col 0 = oldest) into ring-slot order (col 0 = slot 0): logical col j lives at
// slot (ringIdx+j) mod K. A no-op when ringIdx % K == 0 or the K axis broadcasts
// (dim 3 == 1, Q-padding-shaped masks where every key shares one value).
func gatherRingCols(arr *mlx.Array, ringIdx, K int) *mlx.Array {
	if w := arr.Dim(3); w != 1 && w != K {
		panic(fmt.Sprintf("gatherRingCols: K-axis width %d must be 1 or %d", w, K))
	}
	ringIdx %= K
	if ringIdx == 0 || arr.Dim(3) == 1 {
		return arr
	}
	shift := K - ringIdx
	tail := arr.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(shift, K))
	head := arr.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, shift))
	return tail.Concatenate(3, head)
}
