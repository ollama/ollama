package cache

import (
	"slices"

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

	// View returns the current attention history without writing.
	View(b *batch.Batch) *nn.KVHistory
}

type KVCache struct {
	keys, values *mlx.Array
	offset       int
	step         int

	snapshots pendingSnapshots

	// lazySnapshots index into the live keys/values buffer rather than owning a
	// copy (see kvSnapshot); the cache copies them out before overwriting or
	// freeing the slots they name.
	lazySnapshots []*kvSnapshot

	// rewound is set when a restore moves offset backward. The buffer is
	// append-only, so only an append after a rewind can clobber a still-lazy
	// snapshot.
	rewound bool
}

func NewKVCache() *KVCache {
	return &KVCache{step: 256}
}

// Assumes B = 1; heterogeneous batches are not supported.
func (c *KVCache) Update(_ *batch.Batch, keys, values *mlx.Array) *nn.KVHistory {
	start := c.offset
	newK, newV := c.appendKV(keys, values)
	c.captureLazySnapshots(start, c.offset)
	return nn.NewKVHistory(newK, newV, nil)
}

// appendKV is the raw write path shared by Update and Restore.
func (c *KVCache) appendKV(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	B, H, L, Dk, Dv := keys.Dim(0), keys.Dim(1), keys.Dim(2), keys.Dim(3), values.Dim(3)

	prev := c.offset

	// This write fills slots [prev, prev+L). Only an append after a rewind can
	// land on slots a still-lazy snapshot names and overwrite its data, so copy
	// the overlapping snapshots out first. copyOut removes the snapshot from
	// c.lazySnapshots, so range over a clone to avoid skipping entries as the
	// slice shrinks.
	if c.rewound {
		for _, s := range slices.Clone(c.lazySnapshots) {
			if s.fromOffset < prev+L && s.toOffset > prev {
				s.copyOut()
			}
		}
		c.rewound = false
	}

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

// View returns the current cache contents as attention history without writing.
func (c *KVCache) View(_ *batch.Batch) *nn.KVHistory {
	state := c.State()
	return nn.NewKVHistory(state[0], state[1], nil)
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

func (c *KVCache) PrepareSnapshots(offsets []int) { c.snapshots.prepare(c.offset, offsets) }
func (c *KVCache) TakeSnapshots() []Snapshot      { return c.snapshots.take() }

// captureLazySnapshots records edge-local snapshots for the scheduled offsets the
// write [start, end) reached. A KVCache snapshot is a pure index into the
// contiguous append-only buffer, so the write needs no segmenting: one appendKV
// lays down [start, end), then each scheduled offset o captures the edge
// [base, o) by arithmetic, where base is the previous boundary (running cursor).
// Offsets are ascending, so the edges match what segmentation would produce. An
// offset scheduled at start captures a zero-width range and stays nil; rolling
// back there is a live rewind.
func (c *KVCache) captureLazySnapshots(start, end int) {
	for _, o := range c.snapshots.scheduledIn(start, end) {
		c.snapshots.captureReached(o, func(int) Snapshot { return c.lazySnapshot(c.snapshots.base, o) })
	}
}

// kvSnapshot holds paged-out KV data for a range [fromOffset, toOffset).
//
// A snapshot is initially lazy: keys/values are nil and the data lives in the
// issuing cache's buffer at [fromOffset, toOffset). It costs nothing to capture
// and, holding no MLX handle on that buffer, never blocks the in-place append
// donation. The cache copies the range into owned keys/values (copyOut) before
// it overwrites or frees those slots, after which the snapshot is independent
// and cache is nil.
type kvSnapshot struct {
	keys, values         *mlx.Array
	fromOffset, toOffset int
	cache                *KVCache // issuer while lazy; nil once copied out

	// onMaterialize, if set, is fired once from copyOut with the newly-owned
	// byte count so an owner (e.g. the trie's pagedOutBytes counter) can pick
	// up bytes that were free while the snapshot was lazy.
	onMaterialize func(delta int)
}

func (s *kvSnapshot) Size() int {
	if s.keys != nil {
		return s.keys.NumBytes() + s.values.NumBytes()
	}
	// Lazy snapshots own no extra memory: the range still lives in the
	// issuing cache's buffer.
	return 0
}

func (s *kvSnapshot) SetMaterializeHook(fn func(delta int)) { s.onMaterialize = fn }

func (s *kvSnapshot) Close() {
	mlx.Unpin(s.keys, s.values)
	if s.cache != nil {
		s.cache.dropLazySnapshot(s)
		s.cache = nil
	}
}

// copyOut converts a lazy snapshot into an owned [fromOffset, toOffset) copy. It
// is a no-op once the snapshot already owns its data. The copy is an independent
// MLX handle on its own bytes, so a following in-place write to the live buffer
// reallocates rather than mutating data the snapshot still names.
func (s *kvSnapshot) copyOut() {
	if s.keys != nil {
		return
	}
	c := s.cache
	kSlice := c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(s.fromOffset, s.toOffset), mlx.Slice())
	vSlice := c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(s.fromOffset, s.toOffset), mlx.Slice())
	kCopy := mlx.Contiguous(kSlice, false)
	vCopy := mlx.Contiguous(vSlice, false)
	mlx.Pin(kCopy, vCopy)
	mlx.AsyncEval(kCopy, vCopy)

	s.keys, s.values = kCopy, vCopy
	c.dropLazySnapshot(s)
	s.cache = nil

	if s.onMaterialize != nil {
		s.onMaterialize(s.keys.NumBytes() + s.values.NumBytes())
		s.onMaterialize = nil
	}
}

func (c *KVCache) addLazySnapshot(s *kvSnapshot) { c.lazySnapshots = append(c.lazySnapshots, s) }

func (c *KVCache) dropLazySnapshot(s *kvSnapshot) {
	if i := slices.Index(c.lazySnapshots, s); i >= 0 {
		c.lazySnapshots = slices.Delete(c.lazySnapshots, i, i+1)
	}
}

func (c *KVCache) Snapshot(fromOffset int) Snapshot {
	return c.lazySnapshot(fromOffset, c.offset)
}

// lazySnapshot records a lazy [fromOffset, toOffset) snapshot indexing into the
// live buffer. It returns nil for an empty range (the zero-width edge of an
// offset scheduled at the current position).
func (c *KVCache) lazySnapshot(fromOffset, toOffset int) Snapshot {
	if c.keys == nil || toOffset <= fromOffset {
		return nil
	}
	s := &kvSnapshot{
		fromOffset: fromOffset,
		toOffset:   toOffset,
		cache:      c,
	}
	c.addLazySnapshot(s)
	return s
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
		c.rewound = true
		return true
	}

	snap := snapshot.(*kvSnapshot)

	if target > snap.toOffset || c.offset < snap.fromOffset {
		return false
	}

	// A lazy snapshot still in our own set indexes data that is, by construction,
	// still live in our buffer at [fromOffset, toOffset): appendKV copies out any
	// lazy snapshot before overwriting its slots, so one that has stayed lazy was
	// never clobbered.
	if snap.cache == c && snap.keys == nil {
		c.offset = min(target, snap.toOffset)
		c.rewound = true
		return true
	}

	// Own the data before feeding it: appendKV mutates the buffer a lazy snapshot
	// may still index into, so copy out first (no-op if already owned).
	snap.copyOut()

	// Rewind to snapshot start, then feed snapshot. The rewind may expose other
	// outstanding lazy snapshots to the appendKV write, so flag it for the scan.
	c.offset = snap.fromOffset
	c.rewound = true
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

	// Two adjacent lazy snapshots into the same live buffer merge by arithmetic:
	// the combined range [p.from, ch.to) is a single contiguous snapshot, no copy.
	if p.keys == nil && ch.keys == nil && p.cache == ch.cache && p.toOffset == ch.fromOffset {
		merged := &kvSnapshot{fromOffset: p.fromOffset, toOffset: ch.toOffset, cache: p.cache}
		p.cache.addLazySnapshot(merged)
		p.Close()
		ch.Close()
		return merged
	}

	// At least one is an owned copy in its own buffer: concatenate so Restore's
	// single-array appendKV sees one buffer. Own both first.
	p.copyOut()
	ch.copyOut()

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

	// Lazy: split is pure arithmetic into two adjacent lazy snapshots.
	if snap.keys == nil {
		p := &kvSnapshot{fromOffset: snap.fromOffset, toOffset: at, cache: snap.cache}
		ch := &kvSnapshot{fromOffset: at, toOffset: snap.toOffset, cache: snap.cache}
		snap.cache.addLazySnapshot(p)
		snap.cache.addLazySnapshot(ch)
		snap.Close()
		return p, ch
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
	// Freeing drops the buffer every lazy snapshot indexes into; own their data
	// first so they survive independently. copyOut drops the snapshot from
	// c.lazySnapshots, so iterate over a clone to avoid skipping.
	for _, s := range slices.Clone(c.lazySnapshots) {
		s.copyOut()
	}
	mlx.Unpin(c.keys, c.values)
	c.keys, c.values = nil, nil
	c.offset = 0
	c.rewound = false
	c.snapshots = pendingSnapshots{}
}

func (c *KVCache) Offset() int { return c.offset }
