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
// state management. Prepends the prior conv state along axis 1, runs
// the conv, and returns (output, nextConv). nextConv is the trailing
// convTail positions of the concat — write it back to the cache via
// Put alongside the scan's new delta state.
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
func CausalConv1D(b *batch.Batch, input *mlx.Array, conv *Conv1d, weight *mlx.Array, convTail int, opts ...RecurrentOption) (out, nextConv *mlx.Array) {
	cfg := resolveRecurrentConfig(opts)
	var prior *mlx.Array
	if cfg.history != nil {
		prior = cfg.history.ConvState()
	} else {
		prior = cfg.convState
	}

	mask := paddingMask(b, int32(input.Dim(1)))
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
			end := b.SeqQueryLens[i]

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
// Reads prior delta state from the supplied option and returns
// (output, newDelta). Write newDelta back via the cache's Put
// alongside the conv wrapper's nextConv.
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
func GatedDelta(b *batch.Batch, q, k, v, gDecay, beta *mlx.Array, opts ...RecurrentOption) (out, newDelta *mlx.Array) {
	cfg := resolveRecurrentConfig(opts)
	var state *mlx.Array
	if cfg.history != nil {
		state = cfg.history.DeltaState()
	} else {
		state = cfg.deltaState
	}

	return mlx.FastGatedDelta(q, k, v, gDecay, beta, state, paddingMask(b, int32(q.Dim(1))))
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
