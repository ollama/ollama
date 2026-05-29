package cache

import (
	"fmt"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// RotatingKVCache implements sliding window attention with bounded memory
type RotatingKVCache struct {
	maxSize int
	idx     int

	*KVCache
}

func NewRotatingKVCache(maxSize int) *RotatingKVCache {
	return &RotatingKVCache{maxSize: maxSize, KVCache: NewKVCache()}
}

// Assumes B = 1; heterogeneous batches are not supported.
func (c *RotatingKVCache) Update(b *batch.Batch, keys, values *mlx.Array) *nn.KVHistory {
	newK, newV := c.appendKV(keys, values)
	return nn.NewKVHistory(newK, newV, rotatingApplier{
		b:       b,
		K:       newK.Dim(2),
		L:       keys.Dim(2),
		window:  c.maxSize,
		ringIdx: c.idx,
		dtype:   keys.DType(),
	})
}

// View returns the current rotating cache contents in logical order for
// assistant KV sharing.
func (c *RotatingKVCache) View(_ *batch.Batch) *nn.KVHistory {
	k, v := c.logicalTail(c.maxSize - 1)
	if k == nil || v == nil {
		return nil
	}
	return nn.NewKVHistory(k, v, nil)
}

func (c *RotatingKVCache) logicalTail(keep int) (*mlx.Array, *mlx.Array) {
	state := c.State()
	if len(state) < 2 || keep <= 0 {
		return nil, nil
	}

	keys, values := state[0], state[1]
	K := keys.Dim(2)
	if K == 0 {
		return nil, nil
	}

	keep = min(keep, K)
	if K > c.maxSize || c.offset < c.maxSize {
		start := K - keep
		return keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, K), mlx.Slice()),
			values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, K), mlx.Slice())
	}

	oldest := c.idx % K
	var logicalK, logicalV *mlx.Array
	if oldest == 0 {
		logicalK, logicalV = keys, values
	} else {
		tailK := keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(oldest, K), mlx.Slice())
		tailV := values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(oldest, K), mlx.Slice())
		headK := keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, oldest), mlx.Slice())
		headV := values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, oldest), mlx.Slice())
		logicalK = tailK.Concatenate(2, headK)
		logicalV = tailV.Concatenate(2, headV)
	}

	start := K - keep
	return logicalK.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, K), mlx.Slice()),
		logicalV.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, K), mlx.Slice())
}

type speculativeRotatingKVCache struct {
	speculativeBase
	target       *RotatingKVCache
	keys, values *mlx.Array
}

func newSpeculativeRotatingKVCache(target *RotatingKVCache) *speculativeRotatingKVCache {
	return &speculativeRotatingKVCache{
		speculativeBase: speculativeBase{offset: target.Offset()},
		target:          target,
	}
}

func (c *speculativeRotatingKVCache) Update(b *batch.Batch, keys, values *mlx.Array) *nn.KVHistory {
	c.keys = concatKV(c.keys, keys)
	c.values = concatKV(c.values, values)
	c.offset += keys.Dim(2)

	oldK, oldV := c.target.logicalTail(c.target.maxSize - 1)
	histK, histV := c.keys, c.values
	if oldK != nil && oldV != nil {
		histK = oldK.Concatenate(2, c.keys)
		histV = oldV.Concatenate(2, c.values)
	}

	return nn.NewKVHistory(histK, histV, logicalSlidingApplier{
		b:      b,
		K:      histK.Dim(2),
		window: c.target.maxSize,
		dtype:  keys.DType(),
	})
}

func (c *speculativeRotatingKVCache) State() []*mlx.Array {
	if c.keys == nil || c.values == nil {
		return c.target.State()
	}
	oldK, oldV := c.target.logicalTail(c.target.maxSize - 1)
	if oldK == nil || oldV == nil {
		return []*mlx.Array{c.keys, c.values}
	}
	return []*mlx.Array{oldK.Concatenate(2, c.keys), oldV.Concatenate(2, c.values)}
}

func (c *speculativeRotatingKVCache) commit(n int) {
	if c.keys == nil || c.values == nil || n <= 0 {
		return
	}
	n = min(n, c.keys.Dim(2))
	c.target.appendKV(prefixKV(c.keys, n), prefixKV(c.values, n))
}

type logicalSlidingApplier struct {
	b      *batch.Batch
	K      int
	window int
	dtype  mlx.DType
}

func (a logicalSlidingApplier) ApplyMask(logical nn.AttentionMask) nn.AttentionMask {
	return logical.Intersect(nn.SlidingWindowMask(a.b, a.K, a.window, a.dtype))
}

// appendKV is the raw write path shared by Update and Restore —
// routes to concat for prefill (L > 1) and update for decode.
func (c *RotatingKVCache) appendKV(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	if keys.Dim(2) > 1 {
		return c.concat(keys, values)
	}
	return c.update(keys, values)
}

func (c *RotatingKVCache) concat(keys, values *mlx.Array) (newK *mlx.Array, newV *mlx.Array) {
	logutil.Trace("(*RotatingKVCache).concat", "keys_dim", keys.Dims(), "values_dim", values.Dims(), "offset", c.offset, "idx", c.idx, "max_size", c.maxSize)
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
		c.idx = c.keys.Dim(2)
	}

	c.offset += keys.Dim(2)
	c.idx = c.keys.Dim(2)
	return c.keys, c.values
}

func (c *RotatingKVCache) update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	logutil.Trace("(*RotatingKVCache).update", "keys_dim", keys.Dims(), "values_dim", values.Dims(), "offset", c.offset, "idx", c.idx, "max_size", c.maxSize)
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

// rotatingSnapshot holds paged-out data for a RotatingKVCache.
type rotatingSnapshot struct {
	kvSnapshot     // embedded KV data
	idx        int // buffer write position at snapshot time
}

func (s *rotatingSnapshot) Size() int { return s.kvSnapshot.Size() }
func (s *rotatingSnapshot) Close()    { s.kvSnapshot.Close() }

func (c *RotatingKVCache) Snapshot(fromOffset int) Snapshot {
	if c.keys == nil || c.offset <= fromOffset {
		return nil
	}

	state := c.State()
	k := state[0].Clone()
	v := state[1].Clone()
	mlx.Pin(k, v)

	return &rotatingSnapshot{
		kvSnapshot: kvSnapshot{
			keys:       k,
			values:     v,
			fromOffset: fromOffset,
			toOffset:   c.offset,
		},
		idx: c.idx,
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
		// Live rewind is only safe when the buffer hasn't filled yet
		// (offset <= maxSize). Once the window has shifted, rewinding
		// leaves fewer than maxSize trailing tokens to attend to —
		// a snapshot is required to restore the full window.
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

	// Restore from snapshot: rebuild buffer state.
	// Free existing state first.
	if c.keys != nil {
		mlx.Unpin(c.keys, c.values)
	}
	c.keys = snap.keys.Clone()
	c.values = snap.values.Clone()
	mlx.Pin(c.keys, c.values)
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

func (c *RotatingKVCache) Free() {
	c.KVCache.Free()
	c.idx = 0
}

// rotatingApplier composes the sliding-window storage restriction
// onto the caller's logical mask.
//
// ringIdx is the cache's write cursor at Update time. At L=1 decode
// the ring buffer is not position-ordered — logical col j lives at
// storage slot (ringIdx+j) mod K — so tensor masks built in
// logical space must be gathered into this layout before the kernel
// sees them. At L>1 prefill the concat path has already linearised
// storage, so the gather is identity and ringIdx is unused.
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
		// Single-query decode: storage already enforces the window
		// (Update keeps the last maxSize tokens, all within
		// [absQ-window+1, absQ]), and every stored key's absolute
		// position <= absQ. For a zero or plain-causal logical mask
		// both constraints reduce to "no mask", so return the zero
		// mask and let SDPA dispatch to mode="".
		if logical.IsZero() || logical.IsCausal() {
			return nn.AttentionMask{}
		}

		// Tensor-backed mask (user ArrayMask, causal+Relax, causal
		// with accumulated array): materialize in logical-position
		// order then gather K cols into ring-slot order so they
		// align with the cache output the kernel will index.
		arr := logical.AsArray(r.b, r.K, r.dtype)
		arr = gatherRingCols(arr, r.ringIdx, r.K)
		return nn.ArrayMask(arr)
	}

	return logical.Intersect(nn.SlidingWindowMask(r.b, r.K, r.window, r.dtype))
}

// gatherRingCols reorders a [B, 1, L, K] mask's K axis from
// logical-position order (col 0 = oldest stored position) into the
// cache's ring-slot order (col 0 = buffer slot 0). Logical col j
// lives at slot (ringIdx+j) mod K, so storage slot s reads from
// logical col (s-ringIdx+K) mod K. Returns arr unchanged when the
// permutation is a no-op: ringIdx % K == 0 (layouts coincide), or
// the K axis broadcasts (dim 3 == 1, i.e. Q-padding-shaped masks
// where every key shares the same value).
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
