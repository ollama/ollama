package nn

import (
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

func (s seg) len() int32 { return s.end - s.start }

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

// segmentLens returns each row's real query length within segment
// [s.start, s.end): the portion of the row's full real length that falls
// inside the segment, clamped to the segment width. Rows that already ended
// before the segment contribute 0; rows extending past it are full.
func segmentLens(b *batch.Batch, s seg) []int32 {
	lens := make([]int32, len(b.SeqQueryLens))
	for i := range lens {
		lens[i] = min(max(b.SeqQueryLens[i]-s.start, 0), s.len())
	}
	return lens
}

// sliceSeg slices x to the segment's window [s.start, s.end) along the L axis
// (axis 1), keeping all other axes whole. Works for any rank — [B, L],
// [B, L, H], [B, L, H, D] — so the padding mask, conv/gate/beta, and q/k/v all
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
// the conv.
//
// Conv selection: when conv is non-nil (a full nn.Conv1d layer), it
// runs through conv.Forward. Otherwise weight is treated as the bare
// depthwise kernel [C, K] and the fallback manual implementation runs.
// Exactly one of conv or weight should be non-nil.
//
// Shapes: input [B, L, D]; prior state [B, convTail, D]; output
// [B, L, D] (the causal conv strips the prepended state).
//
// Prior state comes from exactly one of WithRecurrentHistory (cache
// path) or WithRecurrentState (no-cache path).
//
// Returns the output and the conv states at each boundary, ending with the
// forward-end conv tail. Without WithSnapshotSplits there is one boundary (the
// end), so states has length 1. With splits, the conv runs in segments and
// states holds the conv tail at each interior split and at the end; out is
// identical to the unsegmented conv.
func CausalConv1D(b *batch.Batch, input *mlx.Array, conv *Conv1d, weight *mlx.Array, convTail int, opts ...RecurrentOption) (out *mlx.Array, states []*mlx.Array) {
	cfg := resolveRecurrentConfig(opts)
	var prior *mlx.Array
	if cfg.history != nil {
		prior = cfg.history.ConvState()
	} else {
		prior = cfg.convState
	}

	L := int32(input.Dim(1))
	mask := paddingMask(b, L) // built once per forward, sliced per segment
	segs := segmentRanges(cfg.splits, L)
	if len(segs) <= 1 {
		out, end := causalConv1DSegment(input, prior, mask, b.SeqQueryLens, conv, weight, convTail)
		return out, []*mlx.Array{end}
	}

	// Segmented conv: each piece prepends the prior piece's conv tail,
	// recording the conv tail at every boundary (interior splits + end).
	outs := make([]*mlx.Array, 0, len(segs))
	states = make([]*mlx.Array, 0, len(segs))
	state := prior
	for _, s := range segs {
		segOut, segNext := causalConv1DSegment(sliceSeg(input, s), state, sliceSeg(mask, s), segmentLens(b, s), conv, weight, convTail)
		outs = append(outs, segOut)
		state = segNext
		states = append(states, segNext)
	}
	return mlx.Concatenate(outs, 1), states
}

// causalConv1DSegment convolves one segment: prepend prior, convolve, return
// (output, conv tail). mask is the [B, L] padding mask for input (nil if no
// row is padded); queryLens is each row's real query length (used to gather
// the per-row conv tail when masking is in play).
func causalConv1DSegment(input, prior *mlx.Array, mask *mlx.Array, queryLens []int32, conv *Conv1d, weight *mlx.Array, convTail int) (out, nextConv *mlx.Array) {
	if mask != nil {
		zero := mlx.FromValue(float32(0)).AsType(input.DType())
		input = mlx.Where(mlx.ExpandDims(mask, 2), input, zero)
	}

	concat := mlx.Concatenate([]*mlx.Array{prior, input}, 1)
	if conv != nil {
		out = conv.Forward(concat)
	} else {
		out = depthwiseCausalConv1d(concat, weight, int32(input.Dim(1)))
	}

	B := int32(concat.Dim(0))
	total := int32(concat.Dim(1))
	D := int32(concat.Dim(2))

	// Gather the tail from each of the non-padded sequence ends
	if mask != nil && convTail > 0 {
		offsets := make([]int32, int(B)*convTail)

		for i := range int(B) {
			end := queryLens[i]

			for k := range convTail {
				offsets[i*convTail+k] = end + int32(k)
			}
		}

		positions := mlx.NewArrayInt32(offsets, []int32{B, int32(convTail), 1})
		nextConv = mlx.TakeAlongAxis(concat, positions, 1)
	} else {
		nextConv = mlx.SliceStartStop(concat,
			[]int32{0, total - int32(convTail), 0},
			[]int32{B, total, D})
	}

	return out, nextConv
}

// depthwiseCausalConv1d implements a depthwise 1D causal convolution
// manually as a sum of kernel-offset multiplies. x has shape
// [B, inLen, C], weight has shape [C, K]; output has shape [B, outLen, C]
// where outLen = inLen - K + 1 (the caller passes outLen to avoid the
// subtraction). Used as the fallback path in CausalConv1D when no
// full Conv1d layer is configured.
func depthwiseCausalConv1d(x, w *mlx.Array, outLen int32) *mlx.Array {
	if x == nil || w == nil {
		return nil
	}
	if w.NumDims() != 2 {
		return nil
	}
	B := int32(x.Dim(0))
	C := int32(w.Dim(0))
	K := int32(w.Dim(1))
	var out *mlx.Array
	for i := range K {
		seg := mlx.SliceStartStop(x, []int32{0, i, 0}, []int32{B, i + outLen, C})
		wi := mlx.SliceStartStop(w, []int32{0, i}, []int32{C, i + 1})
		wi = mlx.Reshape(wi, 1, 1, C)
		term := mlx.Mul(seg, wi)
		if out == nil {
			out = term
		} else {
			out = mlx.Add(out, term)
		}
	}
	return out
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

	needed := false
	for i := range B {
		if in.batch.SeqQueryLens[i] < in.L {
			needed = true
			break
		}
	}
	if !needed {
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
