package nemotronh

import (
	"log/slog"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// convKernel wraps the 1D convolution kernel tensor
type convKernel struct {
	Weight ml.Tensor `gguf:"weight"`
}

// Mamba2 implements the Mamba2 SSM layer for Nemotron-H.
// The forward pass follows llama.cpp's build_mamba2_layer:
// 1. Input projection: zxBCdt = SSMIn @ hidden
// 2. Split: z, xBC, dt
// 3. Concat with conv state, apply SSMConv, save new conv state
// 4. Apply SiLU to convolved xBC
// 5. Split: x, B, C
// 6. Add dt bias
// 7. SSMScan: y = SSMScan(state, x, dt, A, B, C, ids)
// 8. D skip: y = y + x * D
// 9. Swiglu with z: y = z * silu(y)
// 10. Group RMSNorm
// 11. Output projection
type Mamba2 struct {
	SSMIn      *nn.Linear  `gguf:"ssm_in"`     // n_embd → d_in_proj (2*d_inner + 2*n_group*d_state + n_head)
	SSMConv1D  *convKernel `gguf:"ssm_conv1d"` // conv kernel
	SSMConv1DB ml.Tensor   `gguf:"ssm_conv1d.bias"`
	SSMDtB     ml.Tensor   `gguf:"ssm_dt.bias"` // dt bias [n_head]
	SSMA       ml.Tensor   `gguf:"ssm_a"`       // A parameter [1, n_head]
	SSMD       ml.Tensor   `gguf:"ssm_d"`       // D skip connection [1, n_head]
	SSMNorm    *nn.RMSNorm `gguf:"ssm_norm"`    // group norm
	SSMOut     *nn.Linear  `gguf:"ssm_out"`     // output projection
	Layer      int
}

func (m *Mamba2) Forward(ctx ml.Context, hiddenStates ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error) {
	layer := m.Layer
	hiddenDim := hiddenStates.Dim(0)
	nSeqTokens := hiddenStates.Dim(1)
	switch hiddenStates.Dim(2) {
	case 0:
		hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, nSeqTokens, 1)
	case 1:
	default:
		return nil, ErrUnsupportedBatchLayout
	}

	// Nemotron-H is currently clamped to num_parallel=1.
	if cache != nil && cache.IsSupportedForBatch() {
		if cache.numSeqs() != 1 {
			return nil, ErrUnsupportedBatchLayout
		}
		if seqTokens := cache.seqTokens(); seqTokens > 0 && nSeqTokens != seqTokens {
			return nil, ErrUnsupportedBatchLayout
		}
	}
	nSeqs := 1

	dConv := opts.ssmDConv
	dInner := opts.ssmDInner
	dState := opts.ssmDState
	nHead := opts.ssmNHead
	headDim := dInner / nHead
	nGroup := opts.ssmNGroup

	// {n_embd, n_seq_tokens, n_seqs} => {d_in_proj, n_seq_tokens, n_seqs}
	// d_in_proj = 2*d_inner + 2*n_group*d_state + n_head
	zxBCdt := m.SSMIn.Forward(ctx, hiddenStates)

	// Split into z, xBC, dt
	// z: [head_dim, n_head, n_seq_tokens, n_seqs]
	z := zxBCdt.Slice(ctx, 0, 0, dInner, 1)
	z = z.Reshape(ctx, headDim, nHead, nSeqTokens, nSeqs)

	// xBC: [d_inner + 2*n_group*d_state, n_seq_tokens, n_seqs]
	xBCSize := dInner + 2*nGroup*dState
	xBC := zxBCdt.Slice(ctx, 0, dInner, dInner+xBCSize, 1)
	if nSeqTokens == 1 {
		xBC = xBC.Reshape(ctx, xBCSize, 1, nSeqs)
	}

	// dt: [n_head, n_seq_tokens, n_seqs]
	dt := zxBCdt.Slice(ctx, 0, 2*dInner+2*nGroup*dState, 2*dInner+2*nGroup*dState+nHead, 1)
	if nSeqTokens == 1 {
		dt = dt.Reshape(ctx, nHead, 1, nSeqs)
	} else {
		dt = dt.Contiguous(ctx, nHead, nSeqTokens, nSeqs)
	}

	// Get conv state from cache
	convStates, err := cache.ConvState(ctx, layer)
	if err != nil {
		slog.Warn("nemotronh: failed to get conv state, using zeros", "layer", layer, "error", err)
		convStates = ctx.Input().Zeros(ml.DTypeF32, dConv-1, xBCSize, nSeqs)
	}

	// Reshape conv states: [d_conv-1, xBCSize, n_seqs]
	convStates = convStates.Reshape(ctx, dConv-1, xBCSize, nSeqs)

	// For decode (n_seq_tokens == 1), reshape avoids a transpose/contiguous pair.
	var xBCT ml.Tensor
	if nSeqTokens == 1 {
		xBCT = xBC.Reshape(ctx, 1, xBCSize, nSeqs)
	} else {
		// Prefill path: [xBCSize, n_seq_tokens, n_seqs] -> [n_seq_tokens, xBCSize, n_seqs]
		xBCT = xBC.Permute(ctx, 1, 0, 2, 3)
	}

	// Concatenate with conv state: [d_conv-1 + n_seq_tokens, xBCSize, n_seqs]
	convInput := convStates.Concat(ctx, xBCT, 0)

	// Save new conv state (last d_conv-1 columns)
	lastConvStates := convInput.Slice(ctx, 0, nSeqTokens, nSeqTokens+dConv-1, 1)
	cache.UpdateConvState(ctx, layer, lastConvStates)

	// Apply SSM convolution
	xBC = convInput.SSMConv(ctx, m.SSMConv1D.Weight)

	// Add conv bias
	if m.SSMConv1DB != nil {
		xBC = xBC.Add(ctx, m.SSMConv1DB)
	}

	// Apply SiLU
	xBC = xBC.SILU(ctx)

	// Split xBC into x, B, C
	// x: [head_dim, n_head, n_seq_tokens, n_seqs]
	x := xBC.Slice(ctx, 0, 0, dInner, 1)
	x = x.Reshape(ctx, headDim, nHead, nSeqTokens, nSeqs)

	// B: [d_state, n_group, n_seq_tokens, n_seqs]
	B := xBC.Slice(ctx, 0, dInner, dInner+nGroup*dState, 1)
	B = B.Reshape(ctx, dState, nGroup, nSeqTokens, nSeqs)

	// C: [d_state, n_group, n_seq_tokens, n_seqs]
	C := xBC.Slice(ctx, 0, dInner+nGroup*dState, dInner+2*nGroup*dState, 1)
	C = C.Reshape(ctx, dState, nGroup, nSeqTokens, nSeqs)

	// Add dt bias
	dt = dt.Add(ctx, m.SSMDtB)

	// Get SSM state from cache
	state, err := cache.SSMState(ctx, layer, dState, headDim, nHead)
	if err != nil {
		slog.Warn("nemotronh: failed to get SSM state, using zeros", "layer", layer, "error", err)
		state = ctx.Input().Zeros(ml.DTypeF32, dState, headDim, nHead, nSeqs)
	}

	// SSMScan
	// state: [d_state, head_dim, n_head, n_seqs]
	// returns: [head_dim, n_head, n_seq_tokens, n_seqs] concatenated with new state
	ySsm := state.SSMScan(ctx, x, dt, m.SSMA, B, C, cache.slotsTensor())

	// ySsm is a packed 1D buffer: [y (nSeqTokens*headDim*nHead*nSeqs), newState]
	yElems := headDim * nHead * nSeqTokens * nSeqs
	y := ySsm.View(ctx, 0, yElems).Reshape(ctx, headDim, nHead, nSeqTokens, nSeqs)

	stateOffsetBytes := yElems * x.Stride(0)
	stateElems := dState * headDim * nHead * nSeqs
	newState := ySsm.View(ctx, stateOffsetBytes, stateElems)
	newState = newState.Reshape(ctx, dState, headDim, nHead, nSeqs)

	// Update SSM state in cache
	cache.UpdateSSMState(ctx, layer, newState)

	// D skip connection: y = y + x * D
	if m.SSMD != nil {
		// SSMD shape: [1, n_head] -> broadcast to [head_dim, n_head, n_seq_tokens, n_seqs]
		xD := x.Mul(ctx, m.SSMD)
		y = y.Add(ctx, xD)
	}

	// Swiglu with z: y = z * silu(y)
	y = z.SILU(ctx, y)

	// Group RMSNorm
	if m.SSMNorm != nil {
		// Reshape for group norm: [d_inner/n_group, n_group, n_seq_tokens, n_seqs]
		innerPerGroup := dInner / nGroup
		y = y.Reshape(ctx, innerPerGroup, nGroup, nSeqTokens, nSeqs)
		y = m.SSMNorm.Forward(ctx, y, opts.eps)
	}

	// Reshape back to [d_inner, n_seq_tokens, n_seqs]
	y = y.Reshape(ctx, dInner, nSeqTokens, nSeqs)

	// Output projection
	out := m.SSMOut.Forward(ctx, y)

	// Reshape to 2D for consistency with attention output
	return out.Reshape(ctx, out.Dim(0), nSeqTokens*nSeqs), nil
}
