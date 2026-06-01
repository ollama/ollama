package cache

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// RecurrentCache stores state for linear-recurrent layers.
//
// Conv state takes its dtype from the first Get call (the activation dtype).
// Delta state is always float32: the gated-delta recurrent accumulator runs
// for the full sequence length and needs the extra precision regardless of
// activation dtype.
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

func (c *RecurrentCache) setState(old, v *mlx.Array, contiguous bool) *mlx.Array {
	if v == nil || !v.Valid() {
		return old
	}

	if contiguous {
		v = mlx.Contiguous(v, false)
	}
	v = v.Clone()

	mlx.Pin(v)
	mlx.Unpin(old)

	return v
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

// Get returns the current conv/delta state for the SSM layer's read
// phase. On first call it lazy-initializes zero-filled state tensors
// sized from b.InputIDs and dtyped from the caller's activation dtype.
// On subsequent calls it returns the existing state; batch size and
// dtype must match the first call, since recurrent state is cumulative
// and cannot be reshaped without losing history.
func (c *RecurrentCache) Get(b *batch.Batch, dtype mlx.DType) *nn.RecurrentHistory {
	batch := b.InputIDs.Dim(0)
	if batch <= 0 {
		batch = 1
	}

	if c.convState != nil {
		if got := c.convState.Dim(0); got != batch {
			panic(fmt.Sprintf("recurrent cache: batch size changed mid-sequence (have %d, got %d)", got, batch))
		}
		if got := c.convState.DType(); got != dtype {
			panic(fmt.Sprintf("recurrent cache: conv dtype changed mid-sequence (have %v, got %v)", got, dtype))
		}
		return nn.NewRecurrentHistory(c.convState, c.deltaState)
	}

	c.convState = c.setState(nil, mlx.Zeros(dtype, batch, c.convTail, c.convDim), false)
	c.deltaState = c.setState(nil, mlx.Zeros(mlx.DTypeFloat32, batch, c.numVHeads, c.headVDim, c.headKDim), false)
	return nn.NewRecurrentHistory(c.convState, c.deltaState)
}

// Put stores the post-computation conv/delta states for the SSM
// layer's write phase and advances the cache offset by the current
// forward's real token count.
//
// Assumes B = 1; heterogeneous batches are not supported.
func (c *RecurrentCache) Put(b *batch.Batch, newConv, newDelta *mlx.Array) {
	c.convState = c.setState(c.convState, newConv, true)
	c.deltaState = c.setState(c.deltaState, newDelta, false)
	c.offset += int(b.SeqQueryLens[0])
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

	// Recurrent snapshots encode cumulative state up to exactly
	// snap.offset. Target must match — rewinding would leave stale
	// state, and advancing isn't possible without feeding tokens.
	if target != snap.offset {
		return false
	}

	c.convState = c.setState(c.convState, snap.convState, false)
	c.deltaState = c.setState(c.deltaState, snap.deltaState, false)
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
