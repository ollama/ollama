package cache

import (
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// Recurrent is the contract for caches that back recurrent linear-attention layers.
type Recurrent interface {
	Cache
	Get(b *batch.Batch, dtype mlx.DType) *nn.RecurrentHistory
	Put(b *batch.Batch, newConv, newDelta *mlx.Array)
}

// RecurrentRecorder records the per-token scan inputs needed to commit an
// accepted prefix after a speculative recurrent forward.
type RecurrentRecorder interface {
	Record(qkv, q, k, v, gDecay, beta *mlx.Array)
}

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

func (c *RecurrentCache) ensure(batch int, dtype mlx.DType) {
	if batch <= 0 {
		batch = 1
	}

	// Keep the gated-delta recurrent state in float32 even when activations are
	// bf16/fp16. The convolution tail stays in the activation dtype.
	deltaDType := mlx.DTypeFloat32
	needConv := c.convState == nil || !c.convState.Valid() || c.convState.DType() != dtype ||
		c.convState.Dim(0) != batch || c.convState.Dim(1) != c.convTail || c.convState.Dim(2) != c.convDim
	needDelta := c.deltaState == nil || !c.deltaState.Valid() || c.deltaState.DType() != deltaDType ||
		c.deltaState.Dim(0) != batch || c.deltaState.Dim(1) != c.numVHeads || c.deltaState.Dim(2) != c.headVDim || c.deltaState.Dim(3) != c.headKDim
	if !needConv && !needDelta {
		return
	}

	if needConv {
		c.convState = c.setState(c.convState, mlx.Zeros(dtype, batch, c.convTail, c.convDim), false)
	}
	if needDelta {
		c.deltaState = c.setState(c.deltaState, mlx.Zeros(deltaDType, batch, c.numVHeads, c.headVDim, c.headKDim), false)
	}
}

// Get returns the current conv/delta state for the SSM layer's read
// phase. Lazy-initializes zero-filled state tensors using b.InputIDs
// for the batch size; reallocates if the existing state's batch size
// or dtype no longer matches.
func (c *RecurrentCache) Get(b *batch.Batch, dtype mlx.DType) *nn.RecurrentHistory {
	c.ensure(b.InputIDs.Dim(0), dtype)
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

type speculativeRecurrentCache struct {
	speculativeBase
	target *RecurrentCache

	start int

	initialConv  *mlx.Array
	initialDelta *mlx.Array

	qkv, q, k, v, gDecay, beta *mlx.Array
	fullConv, fullDelta        *mlx.Array
	length                     int
}

func newSpeculativeRecurrentCache(target *RecurrentCache) *speculativeRecurrentCache {
	return &speculativeRecurrentCache{
		speculativeBase: speculativeBase{offset: target.Offset()},
		target:          target,
		start:           target.Offset(),
	}
}

func (c *speculativeRecurrentCache) Get(b *batch.Batch, dtype mlx.DType) *nn.RecurrentHistory {
	if c.fullConv != nil && c.fullDelta != nil {
		return nn.NewRecurrentHistory(c.fullConv, c.fullDelta)
	}

	history := c.target.Get(b, dtype)
	if c.initialConv == nil {
		c.initialConv = history.ConvState()
	}
	if c.initialDelta == nil {
		c.initialDelta = history.DeltaState()
	}
	return history
}

func (c *speculativeRecurrentCache) Record(qkv, q, k, v, gDecay, beta *mlx.Array) {
	c.qkv, c.q, c.k, c.v, c.gDecay, c.beta = qkv, q, k, v, gDecay, beta
	if qkv != nil {
		c.length = qkv.Dim(1)
	}
}

func (c *speculativeRecurrentCache) Put(b *batch.Batch, newConv, newDelta *mlx.Array) {
	c.fullConv, c.fullDelta = newConv, newDelta
	c.offset += int(b.SeqQueryLens[0])
}

func (c *speculativeRecurrentCache) State() []*mlx.Array {
	if c.fullConv != nil && c.fullDelta != nil {
		return []*mlx.Array{c.fullConv, c.fullDelta}
	}
	return c.target.State()
}

func (c *speculativeRecurrentCache) commit(n int) {
	if n <= 0 {
		return
	}
	if c.length > 0 && n > c.length {
		n = c.length
	}

	if c.length > 0 && n == c.length && c.fullConv != nil && c.fullDelta != nil {
		c.target.convState = c.target.setState(c.target.convState, c.fullConv, true)
		c.target.deltaState = c.target.setState(c.target.deltaState, c.fullDelta, false)
		c.target.offset = c.start + n
		return
	}

	if c.initialConv == nil || c.initialDelta == nil || c.qkv == nil || c.q == nil || c.k == nil || c.v == nil || c.gDecay == nil || c.beta == nil {
		return
	}

	qkv := sliceSeq(c.qkv, n)
	convConcat := mlx.Concatenate([]*mlx.Array{c.initialConv, qkv}, 1)
	total := convConcat.Dim(1)
	nextConv := convConcat.Slice(mlx.Slice(), mlx.Slice(total-c.target.convTail, total), mlx.Slice())

	_, delta := mlx.FastGatedDelta(
		sliceSeq(c.q, n),
		sliceSeq(c.k, n),
		sliceSeq(c.v, n),
		sliceSeq(c.gDecay, n),
		sliceSeq(c.beta, n),
		c.initialDelta,
		nil,
	)

	c.target.convState = c.target.setState(c.target.convState, nextConv, true)
	c.target.deltaState = c.target.setState(c.target.deltaState, delta, false)
	c.target.offset = c.start + n
}

func sliceSeq(a *mlx.Array, n int) *mlx.Array {
	switch a.NumDims() {
	case 3:
		return a.Slice(mlx.Slice(), mlx.Slice(0, n), mlx.Slice())
	case 4:
		return a.Slice(mlx.Slice(), mlx.Slice(0, n), mlx.Slice(), mlx.Slice())
	default:
		panic("recurrent speculative sequence tensor must be rank 3 or 4")
	}
}
