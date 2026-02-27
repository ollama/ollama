//go:build mlx

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

func (c *RecurrentCache) setStateMaterialized(dst **mlx.Array, v *mlx.Array) {
	if v == nil || !v.Valid() {
		return
	}
	if *dst == v {
		return
	}

	// Break dependency chains so recurrent state does not retain the full
	// per-token compute graph over time.
	snap := mlx.Snapshot(v)
	mlx.Eval(snap)

	old := *dst
	*dst = snap
	mlx.Pin(snap)

	// Drop references to the previous cached state root and transient incoming
	// graph root now that a detached snapshot is retained in cache. Actual
	// cleanup happens at the runner's normal sweep points.
	if old != nil && old != snap {
		mlx.Unpin(old)
	}
	if v != snap && v != old {
		mlx.Unpin(v)
	}
}

func (c *RecurrentCache) setStateRaw(dst **mlx.Array, v *mlx.Array) {
	if v == nil || !v.Valid() {
		return
	}
	if *dst == v {
		return
	}

	old := *dst
	*dst = v
	mlx.Pin(v)
	if old != nil && old != v {
		mlx.Unpin(old)
	}
}

func (c *RecurrentCache) setStateDetached(dst **mlx.Array, v *mlx.Array, ensureContiguous bool) {
	if v == nil || !v.Valid() {
		return
	}
	if *dst == v {
		return
	}

	root := v
	if ensureContiguous {
		root = mlx.Contiguous(v, false)
	}
	detached := mlx.Detach(root)

	old := *dst
	*dst = detached
	mlx.Pin(detached)
	if old != nil && old != detached {
		mlx.Unpin(old)
	}

	// Intentionally do not force-release root/v here. In the fast path, the detached
	// handle aliases the same MLX value and may still be lazily computed. Releasing the
	// source handles can invalidate the cached state before the next eval/sweep point.
}

func snapshotPinned(a *mlx.Array) *mlx.Array {
	if a == nil || !a.Valid() {
		return nil
	}
	snap := mlx.Snapshot(a)
	mlx.Eval(snap)
	mlx.Pin(snap)
	return snap
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
		c.setStateRaw(&c.convState, mlx.Zeros(dtype, batch, c.convTail, c.convDim))
	}
	if needDelta {
		c.setStateRaw(&c.deltaState, mlx.Zeros(dtype, batch, c.numVHeads, c.headVDim, c.headKDim))
	}
}

func (c *RecurrentCache) ConvState(batch int, dtype mlx.DType) *mlx.Array {
	c.ensure(batch, dtype)
	return c.convState
}

func (c *RecurrentCache) SetConvState(v *mlx.Array) {
	c.setStateMaterialized(&c.convState, v)
}

// SetConvStateFast stores conv state without forcing an immediate snapshot/eval.
// Use only for decode hot paths that accept higher transient memory until the next
// sync/sweep point. The conv-state input is usually a slice view, so request a
// compact contiguous copy to avoid pinning the whole source buffer.
func (c *RecurrentCache) SetConvStateFast(v *mlx.Array) {
	c.setStateDetached(&c.convState, v, true)
}

func (c *RecurrentCache) DeltaState(batch int, dtype mlx.DType) *mlx.Array {
	c.ensure(batch, dtype)
	return c.deltaState
}

func (c *RecurrentCache) SetDeltaState(v *mlx.Array) {
	c.setStateMaterialized(&c.deltaState, v)
}

// SetDeltaStateFast stores delta state without forcing an immediate snapshot/eval.
// Use only for decode hot paths that accept higher transient memory until the next
// sync/sweep point.
func (c *RecurrentCache) SetDeltaStateFast(v *mlx.Array) {
	c.setStateDetached(&c.deltaState, v, false)
}

func (c *RecurrentCache) Advance(n int) {
	c.offset += n
}

func (c *RecurrentCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	return keys, values
}

func (c *RecurrentCache) State() (*mlx.Array, *mlx.Array) {
	return c.convState, c.deltaState
}

// Materialize returns the recurrent state roots (conv and delta) held by the cache.
func (c *RecurrentCache) Materialize() []*mlx.Array {
	out := make([]*mlx.Array, 0, 2)
	if c.convState != nil && c.convState.Valid() {
		out = append(out, c.convState)
	}
	if c.deltaState != nil && c.deltaState.Valid() {
		out = append(out, c.deltaState)
	}
	return out
}

func (c *RecurrentCache) CanTrim() bool { return false }

func (c *RecurrentCache) Trim(n int) int {
	// Recurrent state is not directly trimmable. Divergent prefixes must drop the cache.
	_ = n
	return 0
}

func (c *RecurrentCache) Clone() Cache {
	clone := &RecurrentCache{
		offset:     c.offset,
		convTail:   c.convTail,
		convDim:    c.convDim,
		numVHeads:  c.numVHeads,
		headVDim:   c.headVDim,
		headKDim:   c.headKDim,
		convState:  snapshotPinned(c.convState),
		deltaState: snapshotPinned(c.deltaState),
	}
	return clone
}

func (c *RecurrentCache) Free() {
	mlx.Unpin(c.convState, c.deltaState)
	c.convState, c.deltaState = nil, nil
	c.offset = 0
}

func (c *RecurrentCache) Offset() int { return c.offset }
func (c *RecurrentCache) Len() int    { return c.offset }
