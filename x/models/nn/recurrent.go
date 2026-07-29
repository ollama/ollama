package nn

import (
	"slices"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// RecurrentOption configures a call to CausalConv1D or GatedDelta.
type RecurrentOption func(*recurrentConfig)

// recurrentConfig is the resolved set of inputs supplied via
// RecurrentOption. Exactly one of history or (convState/deltaState)
// must be supplied per call.
type recurrentConfig struct {
	history    *RecurrentHistory
	convState  *mlx.Array
	deltaState *mlx.Array
	splits     []int
}

// WithRecurrentHistory supplies a cache's per-layer view of conv and
// delta state. The cache hides any storage layout (per-row, paged,
// gather/scatter) behind the history.
func WithRecurrentHistory(h *RecurrentHistory) RecurrentOption {
	return func(c *recurrentConfig) { c.history = h }
}

// WithRecurrentState supplies explicit conv and delta state tensors
// for the no-cache path. Each wrapper consumes one of the two — pass
// nil for the unused slot when calling only one wrapper.
func WithRecurrentState(convState, deltaState *mlx.Array) RecurrentOption {
	return func(c *recurrentConfig) {
		c.convState = convState
		c.deltaState = deltaState
	}
}

// WithSnapshotSplits requests that the scan run in segments cut at the given
// offsets within this forward (0 < offset < L), capturing the recurrent state
// at each boundary. The wrapper returns those per-boundary states to the
// caller. Offsets must be sorted ascending and strictly interior; out-of-range
// or duplicate offsets are ignored.
func WithSnapshotSplits(offsets []int) RecurrentOption {
	return func(c *recurrentConfig) { c.splits = offsets }
}

// seg is a half-open token range [start, end) within a forward.
type seg struct{ start, end int32 }

// segmentRanges expands the interior cut offsets into consecutive [a,c) ranges
// covering [0, L). Cuts are assumed sorted, deduped, and strictly interior; an
// empty slice yields a single {0, L} segment.
func segmentRanges(splits []int, L int32) []seg {
	segs := make([]seg, 0, len(splits)+1)
	start := int32(0)
	for _, c := range splits {
		segs = append(segs, seg{start, int32(c)})
		start = int32(c)
	}
	return append(segs, seg{start, L})
}

// sliceSeg slices x to the segment's window [s.start, s.end) along the L axis
// (axis 1), keeping all other axes whole. Works for any rank — [B, L],
// [B, L, H], [B, L, H, D] — so the padding mask, gate/beta, and q/k/v all
// slice the same L range and stay aligned. Returns nil when x is nil (the
// no-padding-mask fast path). Slicing the full-forward mask this way yields
// the segment's own mask, so masks are built once per forward and reused
// across segments and layers.
func sliceSeg(x *mlx.Array, s seg) *mlx.Array {
	if x == nil {
		return nil
	}
	rank := x.NumDims()
	start := make([]int32, rank)
	stop := make([]int32, rank)
	for d := range rank {
		start[d], stop[d] = 0, int32(x.Dim(d))
	}
	start[1], stop[1] = s.start, s.end
	return mlx.SliceStartStop(x, start, stop)
}

// resolve applies opts and panics if WithRecurrentHistory and
// WithRecurrentState were combined or neither was supplied.
func resolveRecurrentConfig(opts []RecurrentOption) recurrentConfig {
	var cfg recurrentConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	haveHistory := cfg.history != nil
	haveState := cfg.convState != nil || cfg.deltaState != nil
	if haveHistory && haveState {
		panic("WithRecurrentHistory and WithRecurrentState are mutually exclusive")
	}
	if !haveHistory && !haveState {
		panic("no recurrent state supplied (use WithRecurrentHistory or WithRecurrentState)")
	}

	return cfg
}

// CausalConv1D runs a depthwise causal 1D convolution with recurrent
// state management. Prepends the prior conv state along axis 1 and runs
// conv.Forward over the combined window.
//
// Shapes: input [B, L, D]; prior state [B, convTail, D]; output
// [B, L, D] (the causal conv strips the prepended state).
//
// Prior state comes from exactly one of WithRecurrentHistory (cache
// path) or WithRecurrentState (no-cache path).
//
// Returns the output and the conv states at each boundary, ending with the
// forward-end conv tail. Without WithSnapshotSplits there is one boundary (the
// end), so states has length 1. With splits, the conv still runs as a single
// pass over the whole window and each boundary's conv tail is sliced out of the
// shared input buffer (a boundary state is purely the trailing convTail input
// positions, so no extra conv launch is needed); out is identical to the
// unsegmented conv.
func CausalConv1D(b *batch.Batch, input *mlx.Array, conv *Conv1d, convTail int, opts ...RecurrentOption) (out *mlx.Array, states []*mlx.Array) {
	cfg := resolveRecurrentConfig(opts)
	var prior *mlx.Array
	if cfg.history != nil {
		prior = cfg.history.ConvState()
	} else {
		prior = cfg.convState
	}

	// concat is [prior(convTail); input]; each boundary tail is sliced from it
	// below.
	L := int32(input.Dim(1))
	if mask := paddingMask(b, L); mask != nil {
		zero := mlx.FromValue(float32(0)).AsType(input.DType())
		input = mlx.Where(mlx.ExpandDims(mask, 2), input, zero)
	}
	concat := mlx.Concatenate([]*mlx.Array{prior, input}, 1)
	out = conv.Forward(concat)

	// Snapshot the conv tail at each segment boundary: interior splits in
	// ascending order, then the forward end at L. Each boundary at offset O
	// captures input positions [O-convTail, O) — the trailing convTail rows of
	// the window through token O.
	segs := segmentRanges(cfg.splits, L)
	states = make([]*mlx.Array, 0, len(segs))
	for _, s := range segs {
		st := convStateAt(concat, b.SeqQueryLens, convTail, s.end)
		if L > 1 {
			// Detach the small window from the forward-sized concat; at L==1 it's tiny.
			st = mlx.Contiguous(st, false)
		}
		states = append(states, st)
	}
	return out, states
}

// convStateAt returns the conv state to cache at boundary: the trailing convTail
// input positions ending at boundary, clamped per row to the row's real length so
// a padded row freezes at its real end rather than capturing padding. The prior
// prefixed in concat shifts those positions to columns [boundary, boundary+convTail).
func convStateAt(concat *mlx.Array, queryLens []int32, convTail int, boundary int32) *mlx.Array {
	B := int32(concat.Dim(0))
	D := int32(concat.Dim(2))

	// A row shorter than boundary ends its window at its own real length (inputs
	// are right-padded), so when any row falls short we gather per row instead of
	// one shared slice. boundary itself is still batch-wide — per-sequence
	// boundaries (real batching) are a future change to this gather and the callers.
	clamped := slices.ContainsFunc(queryLens, func(q int32) bool { return boundary > q })

	if clamped && convTail > 0 {
		offsets := make([]int32, int(B)*convTail)
		for i := range int(B) {
			end := min(boundary, queryLens[i])
			for k := range convTail {
				offsets[i*convTail+k] = end + int32(k)
			}
		}
		positions := mlx.NewArrayInt32(offsets, []int32{B, int32(convTail), 1})
		return mlx.TakeAlongAxis(concat, positions, 1)
	}

	return mlx.SliceStartStop(concat,
		[]int32{0, boundary, 0},
		[]int32{B, boundary + int32(convTail), D})
}

// GatedDelta wraps mlx.FastGatedDelta with recurrent state management.
// Reads prior delta state from the supplied option and returns the output
// and the delta states at each boundary. The boundary states end with the
// forward-end state. Without WithSnapshotSplits there is one boundary (the
// end), so states has length 1.
//
// Shape conventions:
//
//	q:     [B, L, numKeyHeads,   headKDim]
//	k:     [B, L, numKeyHeads,   headKDim]
//	v:     [B, L, numValueHeads, headVDim]
//	state: [B, numValueHeads,    headVDim, headKDim]
//
// Prior state comes from exactly one of WithRecurrentHistory (cache
// path) or WithRecurrentState (no-cache path).
//
// When WithSnapshotSplits supplies interior offsets, the scan runs in
// segments cut at those offsets, threading delta state between them; states
// holds the delta state at each interior split and at the end. out is
// identical to the unsegmented scan.
func GatedDelta(b *batch.Batch, q, k, v, gDecay, beta *mlx.Array, opts ...RecurrentOption) (out *mlx.Array, states []*mlx.Array) {
	cfg := resolveRecurrentConfig(opts)
	var prior *mlx.Array
	if cfg.history != nil {
		prior = cfg.history.DeltaState()
	} else {
		prior = cfg.deltaState
	}

	L := int32(q.Dim(1))
	mask := paddingMask(b, L) // built once per forward, sliced per segment
	segs := segmentRanges(cfg.splits, L)
	if len(segs) <= 1 {
		out, end := mlx.FastGatedDelta(q, k, v, gDecay, beta, prior, mask)
		return out, []*mlx.Array{end}
	}

	// Segmented scan: run each [a,c) piece with the prior segment's state,
	// recording the delta state at every boundary (interior splits + end).
	outs := make([]*mlx.Array, 0, len(segs))
	states = make([]*mlx.Array, 0, len(segs))
	state := prior
	for _, seg := range segs {
		segOut, segState := mlx.FastGatedDelta(
			sliceSeg(q, seg), sliceSeg(k, seg), sliceSeg(v, seg),
			sliceSeg(gDecay, seg), sliceSeg(beta, seg),
			state, sliceSeg(mask, seg),
		)
		outs = append(outs, segOut)
		state = segState
		states = append(states, segState)
	}
	return mlx.Concatenate(outs, 1), states
}

// RecurrentHistory is an opaque per-forward view a recurrent cache
// hands to the SSM kernel wrappers — prior conv and delta state
// tensors. Models do not construct this directly; pass it through
// via WithRecurrentHistory, or use WithRecurrentState on the no-cache
// path.
//
// Opaque structure to model code; accessors ConvState/DeltaState
// provide the escape hatch for custom SSM paths.
type RecurrentHistory struct {
	convState, deltaState *mlx.Array
}

// NewRecurrentHistory constructs a RecurrentHistory. Intended for
// cache implementations across packages; model code uses
// WithRecurrentHistory / WithRecurrentState instead.
func NewRecurrentHistory(convState, deltaState *mlx.Array) *RecurrentHistory {
	return &RecurrentHistory{convState: convState, deltaState: deltaState}
}

// ConvState returns the current convolution state tensor.
//
// Last-resort escape hatch for custom SSM paths — may force a slow
// materialization to canonical form depending on the cache's
// internal storage. Prefer CausalConv1D via WithRecurrentHistory.
func (h *RecurrentHistory) ConvState() *mlx.Array { return h.convState }

// DeltaState returns the current delta state tensor.
//
// Last-resort escape hatch for custom SSM paths — may force a slow
// materialization to canonical form depending on the cache's
// internal storage. Prefer GatedDelta via WithRecurrentHistory.
func (h *RecurrentHistory) DeltaState() *mlx.Array { return h.deltaState }

type paddingMaskInputs struct {
	batch *batch.Batch
	L     int32
}

func (in paddingMaskInputs) build() *mlx.Array {
	B := len(in.batch.SeqQueryLens)

	if !slices.ContainsFunc(in.batch.SeqQueryLens, func(q int32) bool { return q < in.L }) {
		return nil
	}

	L := int(in.L)
	vals := make([]bool, B*L)
	for i := range B {
		n := int(in.batch.SeqQueryLens[i])

		base := i * L
		for j := range n {
			vals[base+j] = true
		}
	}

	return mlx.FromValues(vals, B, L)
}

// paddingMask derives a [B, L] bool mask from b.SeqQueryLens for
// right-padded inputs (real tokens at [0, len_i), padding at
// [len_i, L)). Returns nil when b has no rows or every row is full —
// the no-padding fast path that costs nothing extra.
func paddingMask(b *batch.Batch, L int32) *mlx.Array {
	inputs := paddingMaskInputs{batch: b, L: L}
	if cached, ok := b.Memo.Get(inputs); ok {
		return cached.(*mlx.Array)
	}

	mask := inputs.build()
	b.Memo.Put(inputs, mask)

	return mask
}
