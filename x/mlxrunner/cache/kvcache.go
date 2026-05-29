package cache

import (
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// Attention is the contract for caches that back attention layers
// (KVCache, RotatingKVCache).
type Attention interface {
	Cache

	// Update appends (k, v) and returns an opaque nn.KVHistory for
	// this layer's SDPA.
	Update(b *batch.Batch, keys, values *mlx.Array) *nn.KVHistory
}

type KVCache struct {
	keys, values *mlx.Array
	offset       int
	step         int
}

func NewKVCache() *KVCache {
	return &KVCache{step: 256}
}

// Assumes B = 1; heterogeneous batches are not supported.
func (c *KVCache) Update(_ *batch.Batch, keys, values *mlx.Array) *nn.KVHistory {
	newK, newV := c.appendKV(keys, values)
	return nn.NewKVHistory(newK, newV, nil)
}

// View returns the current cache contents as attention history without writing.
func (c *KVCache) View(_ *batch.Batch) *nn.KVHistory {
	state := c.State()
	if len(state) < 2 {
		return nil
	}
	return nn.NewKVHistory(state[0], state[1], nil)
}

// appendKV is the raw write path shared by Update and Restore.
func (c *KVCache) appendKV(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
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

	// Rewind to snapshot start, then feed snapshot.
	c.offset = snap.fromOffset
	c.appendKV(snap.keys, snap.values)

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

type speculativeBase struct {
	offset int
}

func (s *speculativeBase) Free()                                 {}
func (s *speculativeBase) Offset() int                           { return s.offset }
func (s *speculativeBase) Snapshot(int) Snapshot                 { return nil }
func (s *speculativeBase) Restore(Snapshot, int) bool            { return false }
func (s *speculativeBase) Merge(parent, child Snapshot) Snapshot { return nil }
func (s *speculativeBase) Split(snapshot Snapshot, at int) (Snapshot, Snapshot) {
	return nil, snapshot
}

type speculativeKVCache struct {
	speculativeBase
	target *KVCache
	start  int
	end    int
}

func newSpeculativeKVCache(target *KVCache) *speculativeKVCache {
	return &speculativeKVCache{
		speculativeBase: speculativeBase{offset: target.Offset()},
		target:          target,
		start:           target.Offset(),
		end:             target.Offset(),
	}
}

func (c *speculativeKVCache) Update(b *batch.Batch, keys, values *mlx.Array) *nn.KVHistory {
	history := c.target.Update(b, keys, values)
	c.offset = c.target.Offset()
	c.end = c.target.Offset()
	return history
}

func (c *speculativeKVCache) State() []*mlx.Array {
	return c.target.State()
}

func (c *speculativeKVCache) commit(n int) {
	target := max(c.start, c.start+n)
	if target > c.end {
		target = c.end
	}
	c.target.offset = target
	c.offset = target
}

type isolatedKVCache struct {
	speculativeBase
	target       *KVCache
	keys, values *mlx.Array
}

func newIsolatedKVCache(target *KVCache) *isolatedKVCache {
	return &isolatedKVCache{
		speculativeBase: speculativeBase{offset: target.Offset()},
		target:          target,
	}
}

func (c *isolatedKVCache) Update(_ *batch.Batch, keys, values *mlx.Array) *nn.KVHistory {
	c.keys = concatKV(c.keys, keys)
	c.values = concatKV(c.values, values)
	c.offset += keys.Dim(2)

	state := c.target.State()
	if len(state) < 2 {
		return nn.NewKVHistory(c.keys, c.values, nil)
	}
	return nn.NewKVHistory(state[0].Concatenate(2, c.keys), state[1].Concatenate(2, c.values), nil)
}

func (c *isolatedKVCache) State() []*mlx.Array {
	if c.keys == nil || c.values == nil {
		return c.target.State()
	}
	state := c.target.State()
	if len(state) < 2 {
		return []*mlx.Array{c.keys, c.values}
	}
	return []*mlx.Array{
		state[0].Concatenate(2, c.keys),
		state[1].Concatenate(2, c.values),
	}
}
