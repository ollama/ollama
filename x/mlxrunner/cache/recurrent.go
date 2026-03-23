package cache

import "github.com/ollama/ollama/x/mlxrunner/mlx"

// RecurrentCache stores state for linear-recurrent layers.
//
// Conv state shape: [B, convTail, convDim]
// Delta state shape: [B, numVHeads, headVDim, headKDim]
type RecurrentCache struct {
	convState  *mlx.Array
	deltaState *mlx.Array
	offset     int

	convTail  int
	convDim   int
	numVHeads int
	headVDim  int
	headKDim  int
}

func (c *RecurrentCache) setStateRaw(old, v *mlx.Array) *mlx.Array {
	if v == nil || !v.Valid() {
		return old
	}
	if old == v {
		return old
	}

	mlx.Pin(v)
	if old != nil && old != v {
		mlx.Unpin(old)
	}

	return v
}

func (c *RecurrentCache) setStateDetached(old, v *mlx.Array, ensureContiguous bool) *mlx.Array {
	if v == nil || !v.Valid() {
		return old
	}
	if old == v {
		return old
	}

	root := v
	if ensureContiguous {
		root = mlx.Contiguous(v, false)
	}
	detached := root.Clone()

	mlx.Pin(detached)
	if old != nil && old != detached {
		mlx.Unpin(old)
	}

	return detached
}

func NewRecurrentCache(convTail, convDim, numVHeads, headVDim, headKDim int32) *RecurrentCache {
	return &RecurrentCache{
		convTail:  int(convTail),
		convDim:   int(convDim),
		numVHeads: int(numVHeads),
		headVDim:  int(headVDim),
		headKDim:  int(headKDim),
	}
}

func (c *RecurrentCache) ensure(batch int, dtype mlx.DType) {
	if batch <= 0 {
		batch = 1
	}

	needConv := c.convState == nil || !c.convState.Valid() || c.convState.DType() != dtype ||
		c.convState.Dim(0) != batch || c.convState.Dim(1) != c.convTail || c.convState.Dim(2) != c.convDim
	needDelta := c.deltaState == nil || !c.deltaState.Valid() || c.deltaState.DType() != dtype ||
		c.deltaState.Dim(0) != batch || c.deltaState.Dim(1) != c.numVHeads || c.deltaState.Dim(2) != c.headVDim || c.deltaState.Dim(3) != c.headKDim
	if !needConv && !needDelta {
		return
	}

	if needConv {
		c.convState = c.setStateRaw(c.convState, mlx.Zeros(dtype, batch, c.convTail, c.convDim))
	}
	if needDelta {
		c.deltaState = c.setStateRaw(c.deltaState, mlx.Zeros(dtype, batch, c.numVHeads, c.headVDim, c.headKDim))
	}
}

func (c *RecurrentCache) ConvState(batch int, dtype mlx.DType) *mlx.Array {
	c.ensure(batch, dtype)
	return c.convState
}

func (c *RecurrentCache) SetConvState(v *mlx.Array) {
	c.convState = c.setStateDetached(c.convState, v, true)
}

func (c *RecurrentCache) DeltaState(batch int, dtype mlx.DType) *mlx.Array {
	c.ensure(batch, dtype)
	return c.deltaState
}

func (c *RecurrentCache) SetDeltaState(v *mlx.Array) {
	c.deltaState = c.setStateDetached(c.deltaState, v, false)
}

func (c *RecurrentCache) Advance(n int) {
	c.offset += n
}

func (c *RecurrentCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	return keys, values
}

func (c *RecurrentCache) State() []*mlx.Array {
	return []*mlx.Array{c.convState, c.deltaState}
}

// recurrentSnapshot holds paged-out recurrent state. Self-contained —
// does not depend on any parent state.
type recurrentSnapshot struct {
	convState, deltaState *mlx.Array
	offset                int
}

func (s *recurrentSnapshot) Size() int { return s.convState.NumBytes() + s.deltaState.NumBytes() }
func (s *recurrentSnapshot) Close()    { mlx.Unpin(s.convState, s.deltaState) }

func (c *RecurrentCache) Snapshot(fromOffset int) Snapshot {
	// Recurrent state is not position-sliceable — always snapshot the full state.
	if c.convState == nil && c.deltaState == nil {
		return nil
	}

	snap := &recurrentSnapshot{offset: c.offset}
	snap.convState = c.convState.Clone()
	snap.deltaState = c.deltaState.Clone()
	mlx.Pin(snap.convState, snap.deltaState)

	return snap
}

func (c *RecurrentCache) Restore(snapshot Snapshot, target int) bool {
	if snapshot == nil {
		// Recurrent state is cumulative and can't rewind. Only succeed
		// if we're already at the target (no-op).
		return target == c.offset
	}

	snap := snapshot.(*recurrentSnapshot)

	// Recurrent state encodes all tokens up to snap.offset. Restoring
	// to a target before that would leave stale state from tokens
	// [target, snap.offset) baked in. Only allow restoring forward.
	if target < snap.offset {
		return false
	}

	c.convState = c.setStateRaw(c.convState, snap.convState)
	c.deltaState = c.setStateRaw(c.deltaState, snap.deltaState)
	c.offset = snap.offset

	return true
}

func (c *RecurrentCache) Merge(parent, child Snapshot) Snapshot {
	// Recurrent snapshots are self-contained — child supersedes parent.
	if parent != nil {
		parent.Close()
	}
	return child
}

func (c *RecurrentCache) Split(snapshot Snapshot, at int) (Snapshot, Snapshot) {
	// Recurrent state is cumulative and not position-sliceable.
	// Cannot recover intermediate state at the split point.
	return nil, snapshot
}

func (c *RecurrentCache) Free() {
	mlx.Unpin(c.convState, c.deltaState)
	c.convState, c.deltaState = nil, nil
	c.offset = 0
}

func (c *RecurrentCache) Offset() int { return c.offset }
