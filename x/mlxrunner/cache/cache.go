package cache

import (
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type Cache interface {
	Update(keys, values *mlx.Array) (newKeys, newValues *mlx.Array)
	// State returns the cache-owned state roots that should be kept/evaluated.
	State() []*mlx.Array
	Free()
	Offset() int

	// Snapshot copies cache state from fromOffset to current offset into
	// pinned VRAM arrays. The active cache is unchanged.
	Snapshot(fromOffset int) Snapshot

	// Restore brings the cache to target. If snapshot is nil, rewinds
	// using the cache's own live state. Returns false if the target is
	// unreachable (e.g. target > current offset, or negative).
	Restore(snapshot Snapshot, target int) bool

	// Merge combines two sequential snapshots [a,b) and [b,c) into [a,c).
	// Takes ownership of both inputs.
	Merge(parent, child Snapshot) Snapshot

	// Split divides a snapshot [a,c) at offset b into [a,b) and [b,c).
	// Takes ownership of the input. Cache types that cannot split
	// (e.g. recurrent) return (nil, snapshot).
	Split(snapshot Snapshot, at int) (parent, child Snapshot)
}

// Snapshot is paged-out cache state that can be restored later.
type Snapshot interface {
	// Size returns the byte size of the paged-out data (in VRAM).
	Size() int
	// Close unpins the snapshot's arrays so they can be freed by Sweep.
	Close()
}

type KVCache struct {
	keys, values *mlx.Array
	offset       int
	step         int
}

func NewKVCache() *KVCache {
	return &KVCache{step: 256}
}

func (c *KVCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	B, H, L, Dk, Dv := keys.Dim(0), keys.Dim(1), keys.Dim(2), keys.Dim(3), values.Dim(3)

	prev := c.offset

	// Grow buffer if needed
	if c.keys == nil || (prev+L) > c.keys.Dim(2) {
		steps := (c.step + L - 1) / c.step
		newKeys := mlx.Zeros(keys.DType(), B, H, steps*c.step, Dk)
		newValues := mlx.Zeros(values.DType(), B, H, steps*c.step, Dv)

		if c.keys != nil {
			if prev%c.step != 0 {
				c.keys.Set(c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, prev), mlx.Slice()))
				c.values.Set(c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, prev), mlx.Slice()))
			}
			c.keys.Set(c.keys.Concatenate(2, newKeys))
			c.values.Set(c.values.Concatenate(2, newValues))
		} else {
			c.keys, c.values = newKeys, newValues
			mlx.Pin(c.keys, c.values)
		}
	}

	c.offset += L
	c.keys.Set(c.keys.SliceUpdate(keys, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, c.offset), mlx.Slice()))
	c.values.Set(c.values.SliceUpdate(values, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, c.offset), mlx.Slice()))

	return c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice())
}

func (c *KVCache) State() []*mlx.Array {
	if c.keys == nil || c.values == nil {
		return nil
	}
	return []*mlx.Array{
		c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice()),
	}
}

// kvSnapshot holds paged-out KV data for a range [fromOffset, toOffset).
type kvSnapshot struct {
	keys, values         *mlx.Array
	fromOffset, toOffset int
}

func (s *kvSnapshot) Size() int { return s.keys.NumBytes() + s.values.NumBytes() }
func (s *kvSnapshot) Close()    { mlx.Unpin(s.keys, s.values) }

func (c *KVCache) Snapshot(fromOffset int) Snapshot {
	if c.keys == nil || c.offset <= fromOffset {
		return nil
	}
	from := max(0, fromOffset)
	to := c.offset

	kSlice := c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(from, to), mlx.Slice())
	vSlice := c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(from, to), mlx.Slice())
	kCopy := mlx.Contiguous(kSlice, false)
	vCopy := mlx.Contiguous(vSlice, false)
	mlx.Pin(kCopy, vCopy)
	mlx.AsyncEval(kCopy, vCopy)

	return &kvSnapshot{
		keys:       kCopy,
		values:     vCopy,
		fromOffset: from,
		toOffset:   to,
	}
}

func (c *KVCache) Restore(snapshot Snapshot, target int) bool {
	if target < 0 {
		return false
	}

	if snapshot == nil {
		if target > c.offset {
			return false
		}
		c.offset = target
		return true
	}

	snap := snapshot.(*kvSnapshot)

	if target > snap.toOffset || c.offset < snap.fromOffset {
		return false
	}

	// Rewind to snapshot start, then feed snapshot data through Update.
	c.offset = snap.fromOffset
	c.Update(snap.keys, snap.values)

	// Clamp to target if needed (target may be less than full snapshot).
	if target < c.offset {
		c.offset = target
	}

	return true
}

func (c *KVCache) Merge(parent, child Snapshot) Snapshot {
	if parent == nil || child == nil {
		if parent != nil {
			parent.Close()
		}
		if child != nil {
			child.Close()
		}
		return nil
	}
	p := parent.(*kvSnapshot)
	ch := child.(*kvSnapshot)

	mk := p.keys.Concatenate(2, ch.keys)
	mv := p.values.Concatenate(2, ch.values)
	mlx.Pin(mk, mv)
	mlx.AsyncEval(mk, mv)

	p.Close()
	ch.Close()

	return &kvSnapshot{
		keys:       mk,
		values:     mv,
		fromOffset: p.fromOffset,
		toOffset:   ch.toOffset,
	}
}

func (c *KVCache) Split(snapshot Snapshot, at int) (Snapshot, Snapshot) {
	if snapshot == nil {
		return nil, nil
	}
	snap := snapshot.(*kvSnapshot)
	splitIdx := at - snap.fromOffset
	seqLen := snap.toOffset - snap.fromOffset
	if splitIdx <= 0 {
		return nil, snapshot
	}
	if splitIdx >= seqLen {
		return snapshot, nil
	}

	pk := mlx.Contiguous(snap.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, splitIdx), mlx.Slice()), false)
	pv := mlx.Contiguous(snap.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, splitIdx), mlx.Slice()), false)
	ck := mlx.Contiguous(snap.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(splitIdx, seqLen), mlx.Slice()), false)
	cv := mlx.Contiguous(snap.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(splitIdx, seqLen), mlx.Slice()), false)
	mlx.Pin(pk, pv, ck, cv)
	mlx.AsyncEval(pk, pv, ck, cv)

	snap.Close()

	p := &kvSnapshot{
		keys:       pk,
		values:     pv,
		fromOffset: snap.fromOffset,
		toOffset:   at,
	}
	ch := &kvSnapshot{
		keys:       ck,
		values:     cv,
		fromOffset: at,
		toOffset:   snap.toOffset,
	}
	return p, ch
}

func (c *KVCache) Free() {
	mlx.Unpin(c.keys, c.values)
	c.keys, c.values = nil, nil
	c.offset = 0
}

func (c *KVCache) Offset() int { return c.offset }

// RotatingKVCache implements sliding window attention with bounded memory
type RotatingKVCache struct {
	maxSize int
	idx     int

	*KVCache
}

func NewRotatingKVCache(maxSize int) *RotatingKVCache {
	return &RotatingKVCache{maxSize: maxSize, KVCache: NewKVCache()}
}

func (c *RotatingKVCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
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
			c.keys.Set(c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.idx), mlx.Slice()))
			c.values.Set(c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.idx), mlx.Slice()))
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
	return []*mlx.Array{
		c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice()),
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
