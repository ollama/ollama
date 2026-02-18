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

	// Release previous cached state root, then recursively free the transient
	// incoming graph root now that a detached snapshot is retained in cache.
	if old != nil && old != snap {
		mlx.Release(old)
	}
	if v != snap && v != old {
		mlx.Free(v)
	}
}

func (c *RecurrentCache) setStateRaw(dst **mlx.Array, v *mlx.Array) {
	if v == nil || !v.Valid() {
		return
	}
	old := *dst
	*dst = v
	if old != nil && old != v {
		mlx.Release(old)
	}
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

	if c.convState == nil || c.convState.DType() != dtype ||
		c.convState.Dim(0) != batch || c.convState.Dim(1) != c.convTail || c.convState.Dim(2) != c.convDim {
		c.setStateRaw(&c.convState, mlx.Zeros(dtype, batch, c.convTail, c.convDim))
	}

	if c.deltaState == nil || c.deltaState.DType() != dtype ||
		c.deltaState.Dim(0) != batch || c.deltaState.Dim(1) != c.numVHeads || c.deltaState.Dim(2) != c.headVDim || c.deltaState.Dim(3) != c.headKDim {
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

func (c *RecurrentCache) DeltaState(batch int, dtype mlx.DType) *mlx.Array {
	c.ensure(batch, dtype)
	return c.deltaState
}

func (c *RecurrentCache) SetDeltaState(v *mlx.Array) {
	c.setStateMaterialized(&c.deltaState, v)
}

func (c *RecurrentCache) Advance(n int) {
	c.offset += n
}

func (c *RecurrentCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	return keys, values
}

func (c *RecurrentCache) State() (*mlx.Array, *mlx.Array) {
	c.ensure(1, mlx.DTypeFloat32)
	return c.convState, c.deltaState
}

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

func (c *RecurrentCache) Trim(n int) int {
	n = min(c.offset, n)
	c.offset -= n
	// Recurrent state cannot be reversed cheaply; reset to a clean state when trimming.
	if n > 0 {
		if c.convState != nil {
			c.setStateRaw(&c.convState, mlx.Zeros(c.convState.DType(), c.convState.Dim(0), c.convState.Dim(1), c.convState.Dim(2)))
		}
		if c.deltaState != nil {
			c.setStateRaw(&c.deltaState, mlx.Zeros(c.deltaState.DType(), c.deltaState.Dim(0), c.deltaState.Dim(1), c.deltaState.Dim(2), c.deltaState.Dim(3)))
		}
	}
	return n
}

func (c *RecurrentCache) Clone() Cache {
	clone := &RecurrentCache{
		offset:    c.offset,
		convTail:  c.convTail,
		convDim:   c.convDim,
		numVHeads: c.numVHeads,
		headVDim:  c.headVDim,
		headKDim:  c.headKDim,
	}
	if c.convState != nil {
		clone.convState = c.convState.Clone()
	}
	if c.deltaState != nil {
		clone.deltaState = c.deltaState.Clone()
	}
	return clone
}

func (c *RecurrentCache) Offset() int { return c.offset }
func (c *RecurrentCache) Len() int    { return c.offset }
