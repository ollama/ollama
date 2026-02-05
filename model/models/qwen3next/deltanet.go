package qwen3next

import (
	"errors"
	"log/slog"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

const chunkSize = 64

// TriType constants for triangular matrix operations
const (
	TriTypeUpperDiag = 0
	TriTypeUpper     = 1
	TriTypeLowerDiag = 2
	TriTypeLower     = 3
)

// convKernel wraps the 1D convolution kernel tensor
type convKernel struct {
	Weight ml.Tensor `gguf:"weight"`
}

// Masks holds pre-computed mask tensors for chunked attention
type Masks struct {
	Causal   ml.Tensor // Lower triangular [chunkSize, chunkSize]
	Identity ml.Tensor // Diagonal [chunkSize, chunkSize]
	Diag     ml.Tensor // causal + identity
}

// GatedDeltaNet implements linear attention with SSM convolution and recurrent state.
// It implements the Operator interface directly.
type GatedDeltaNet struct {
	// Optimized path: pre-split QKV and gate
	SSMQKV       *nn.Linear  `gguf:"attn_qkv"`  // -> Q, K, V (concatenated)
	SSMQKVGate   *nn.Linear  `gguf:"attn_gate"` // -> Z gate
	SSMBetaAlpha *nn.Linear  `gguf:"ssm_ba"`    // -> beta, alpha
	SSMConv1D    *convKernel `gguf:"ssm_conv1d"`
	SSMDT        ml.Tensor   `gguf:"ssm_dt"` // alpha bias
	SSMA         ml.Tensor   `gguf:"ssm_a"`  // -A_log.exp()
	SSMNorm      *nn.RMSNorm `gguf:"ssm_norm"`
	SSMOut       *nn.Linear  `gguf:"ssm_out"`

	// Layer index for cache access (set during model construction)
	Layer int
}

type stateAccessors struct {
	convState   func() (ml.Tensor, error)
	updateConv  func(ml.Tensor)
	deltaState  func() (ml.Tensor, error)
	updateDelta func(ml.Tensor)
}

// createMasks builds the constant mask tensors (called once, reused for all chunks)
func createMasks(ctx ml.Context) *Masks {
	ones := ctx.Input().Zeros(ml.DTypeF32, chunkSize, chunkSize)
	ones = ones.Fill(ctx, 1.0)
	causalMask := ones.Tri(ctx, TriTypeLower)

	onesVec := ctx.Input().Zeros(ml.DTypeF32, chunkSize)
	onesVec = onesVec.Fill(ctx, 1.0)
	identity := onesVec.Diag(ctx)

	diagMask := causalMask.Add(ctx, identity)

	return &Masks{
		Causal:   causalMask,
		Identity: identity,
		Diag:     diagMask,
	}
}

func (gdn *GatedDeltaNet) Forward(ctx ml.Context, hiddenStates, _ ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error) {
	nSeqTokens := hiddenStates.Dim(1)
	nSeqs := hiddenStates.Dim(2)
	if cache != nil && cache.IsSupportedForBatch() {
		seqTokens := cache.seqTokens()
		seqs := cache.numSeqs()
		if seqTokens > 0 && seqs > 0 {
			if nSeqs > 1 {
				if nSeqTokens != seqTokens || nSeqs != seqs {
					return gdn.forwardMixed(ctx, hiddenStates, cache, opts)
				}
			} else {
				if nSeqTokens != seqTokens*seqs {
					return gdn.forwardMixed(ctx, hiddenStates, cache, opts)
				}
				hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), seqTokens, seqs)
			}
		}

		numVHeads := opts.ssmDtRank
		headVDim := opts.ssmDInner / numVHeads
		layer := gdn.Layer
		access := stateAccessors{
			convState: func() (ml.Tensor, error) {
				return cache.ConvState(ctx, layer)
			},
			updateConv: func(newState ml.Tensor) {
				cache.UpdateConvState(ctx, layer, newState)
			},
			deltaState: func() (ml.Tensor, error) {
				return cache.DeltaState(ctx, layer, headVDim, numVHeads)
			},
			updateDelta: func(newState ml.Tensor) {
				cache.UpdateDeltaState(ctx, layer, newState)
			},
		}

		return gdn.forwardWithAccessors(ctx, hiddenStates, opts, access)
	}

	if cache == nil {
		return nil, ErrUnsupportedBatchLayout
	}

	return gdn.forwardMixed(ctx, hiddenStates, cache, opts)
}

func (gdn *GatedDeltaNet) forwardMixed(ctx ml.Context, hiddenStates ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error) {
	if hiddenStates.Dim(2) > 0 {
		hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), hiddenStates.Dim(1)*hiddenStates.Dim(2))
	}

	if len(cache.curSeqs) == 0 {
		return hiddenStates, nil
	}

	// Ensure any shared slots are detached once for this forward pass.
	cache.ensureWritableOnce(ctx)

	layer := gdn.Layer
	numVHeads := opts.ssmDtRank
	headVDim := opts.ssmDInner / numVHeads

	if gdn.SSMQKV == nil || gdn.SSMQKVGate == nil {
		return nil, errors.New("qwen3next: missing attn_qkv/attn_gate projections (legacy ssm_in is not supported)")
	}

	// Precompute projections for the full batch and slice per sequence.
	mixedBAFull := gdn.SSMBetaAlpha.Forward(ctx, hiddenStates)
	qkvMixedFull := gdn.SSMQKV.Forward(ctx, hiddenStates)
	zFull := gdn.SSMQKVGate.Forward(ctx, hiddenStates)

	out := hiddenStates
	for seqIndex := range cache.curSeqs {
		idxs := cache.curSeqTokenIdxs[seqIndex]
		if len(idxs) == 0 {
			continue
		}
		idxTensor := ctx.Input().FromInts(idxs, len(idxs))

		mixedBA := mixedBAFull.Rows(ctx, idxTensor)
		qkvMixed := qkvMixedFull.Rows(ctx, idxTensor)
		z := zFull.Rows(ctx, idxTensor)

		slot := cache.curSlots[seqIndex]
		access := stateAccessors{
			convState: func() (ml.Tensor, error) {
				return cache.convStateForSlot(ctx, layer, slot)
			},
			updateConv: func(newState ml.Tensor) {
				cache.updateConvStateForSlot(ctx, layer, slot, seqIndex, newState)
			},
			deltaState: func() (ml.Tensor, error) {
				return cache.deltaStateForSlot(ctx, layer, slot, headVDim, numVHeads)
			},
			updateDelta: func(newState ml.Tensor) {
				cache.updateDeltaStateForSlot(ctx, layer, slot, seqIndex, newState)
			},
		}

		seqOut, err := gdn.forwardProjected(ctx, len(idxs), 1, mixedBA, qkvMixed, z, opts, access)
		if err != nil {
			return nil, err
		}
		out = out.SetRows(ctx, seqOut, idxTensor)
	}

	return out, nil
}

func (gdn *GatedDeltaNet) forwardWithAccessors(ctx ml.Context, hiddenStates ml.Tensor, opts *Options, access stateAccessors) (ml.Tensor, error) {
	nSeqTokens := hiddenStates.Dim(1)
	nSeqs := hiddenStates.Dim(2)

	mixedBA := gdn.SSMBetaAlpha.Forward(ctx, hiddenStates)

	if gdn.SSMQKV == nil || gdn.SSMQKVGate == nil {
		return nil, errors.New("qwen3next: missing attn_qkv/attn_gate projections (legacy ssm_in is not supported)")
	}
	// Optimized path: pre-split QKV and gate
	qkvMixed := gdn.SSMQKV.Forward(ctx, hiddenStates)
	z := gdn.SSMQKVGate.Forward(ctx, hiddenStates)

	return gdn.forwardProjected(ctx, nSeqTokens, nSeqs, mixedBA, qkvMixed, z, opts, access)
}

func (gdn *GatedDeltaNet) forwardProjected(
	ctx ml.Context,
	nSeqTokens, nSeqs int,
	mixedBA, qkvMixed, z ml.Tensor,
	opts *Options,
	access stateAccessors,
) (ml.Tensor, error) {
	layer := gdn.Layer

	headKDim := opts.ssmDState
	numKHeads := opts.ssmNGroup
	numVHeads := opts.ssmDtRank
	headVDim := opts.ssmDInner / numVHeads
	convKernelSize := opts.convKernelSize
	qkvDim := headKDim*numKHeads*2 + headVDim*numVHeads

	qkvMixed = qkvMixed.Reshape(ctx, qkvDim, nSeqTokens, nSeqs)

	baNewDim := 2 * numVHeads / numKHeads
	mixedBAReshaped := mixedBA.Reshape(ctx, baNewDim, numKHeads, nSeqTokens, nSeqs)

	// Split beta and alpha
	betaSize := numVHeads / numKHeads
	alphaSize := numVHeads / numKHeads

	b := mixedBAReshaped.Slice(ctx, 0, 0, betaSize, 1)
	a := mixedBAReshaped.Slice(ctx, 0, betaSize, betaSize+alphaSize, 1)

	// Reshape to merge head dimensions
	beta := b.Contiguous(ctx, numVHeads, 1, nSeqTokens, nSeqs)
	alpha := a.Contiguous(ctx, numVHeads, nSeqTokens, nSeqs)

	// Compute gate: softplus(alpha + dt_bias) * -A
	alphaBiased := alpha.Add(ctx, gdn.SSMDT)
	alphaSoftplus := alphaBiased.Softplus(ctx)
	gate := alphaSoftplus.Mul(ctx, gdn.SSMA)
	qkvMixed = qkvMixed.Permute(ctx, 1, 0, 2, 3)

	// Get conv state from cache
	convStates, err := access.convState()
	if err != nil {
		// Log this - if it happens, short-term context will be lost
		slog.Warn("qwen3next: failed to get conv state, using zeros", "layer", layer, "error", err)
		convStates = ctx.Input().Zeros(ml.DTypeF32, convKernelSize-1, qkvDim, nSeqs)
	}

	// Reshape conv states
	convStates = convStates.Reshape(ctx, convKernelSize-1, qkvDim, nSeqs)

	// Concatenate with input for convolution
	convInput := convStates.Concat(ctx, qkvMixed, 0)

	// Save new conv state (last convKernelSize-1 tokens)
	lastConvStates := convInput.Slice(ctx, 0, nSeqTokens, nSeqTokens+convKernelSize-1, 1)
	access.updateConv(lastConvStates)

	// Apply SSM convolution (kernel must be F32 for Metal)
	convOutput := convInput.SSMConv(ctx, gdn.SSMConv1D.Weight)
	convOutput = convOutput.SILU(ctx)

	// Reshape for extraction
	convQKVMix := convOutput.Contiguous(ctx, qkvDim, nSeqTokens*nSeqs)

	// Extract convolved Q, K, V
	qConv := convQKVMix.Slice(ctx, 0, 0, headKDim*numKHeads, 1)
	kConv := convQKVMix.Slice(ctx, 0, headKDim*numKHeads, 2*headKDim*numKHeads, 1)
	vConv := convQKVMix.Slice(ctx, 0, 2*headKDim*numKHeads, qkvDim, 1)

	// Reshape to 4D
	qConv = qConv.Contiguous(ctx, headKDim, numKHeads, nSeqTokens, nSeqs)
	kConv = kConv.Contiguous(ctx, headKDim, numKHeads, nSeqTokens, nSeqs)
	vConv = vConv.Contiguous(ctx, headVDim, numVHeads, nSeqTokens, nSeqs)

	// Get delta state from cache
	state, err := access.deltaState()
	if err != nil {
		// Log this - if it happens frequently, context will degrade
		slog.Warn("qwen3next: failed to get delta state, using zeros", "layer", layer, "error", err)
		state = ctx.Input().Zeros(ml.DTypeF32, headVDim, headVDim*numVHeads, nSeqs)
	}
	state = state.Reshape(ctx, headVDim, headVDim*numVHeads, 1, nSeqs)

	// Repeat interleave Q and K if numKHeads != numVHeads
	if numKHeads != numVHeads {
		repeatFactor := numVHeads / numKHeads

		qReshaped := qConv.Reshape(ctx, headKDim, 1, numKHeads*nSeqTokens*nSeqs)
		kReshaped := kConv.Reshape(ctx, headKDim, 1, numKHeads*nSeqTokens*nSeqs)

		qRepeated := qReshaped.Repeat4D(ctx, headKDim, repeatFactor, numKHeads*nSeqTokens*nSeqs, 1)
		kRepeated := kReshaped.Repeat4D(ctx, headKDim, repeatFactor, numKHeads*nSeqTokens*nSeqs, 1)

		qConv = qRepeated.Reshape(ctx, headKDim, numKHeads*repeatFactor, nSeqTokens, nSeqs)
		kConv = kRepeated.Reshape(ctx, headKDim, numKHeads*repeatFactor, nSeqTokens, nSeqs)
	}

	// Choose computation mode based on sequence length
	var (
		attnOut  ml.Tensor
		newState ml.Tensor
	)
	if nSeqTokens == 1 {
		attnOut, newState = gdn.deltaNetAutoregressive(ctx, qConv, kConv, vConv, gate, beta, state, opts)
	} else {
		// Use pre-computed masks from opts (created once in Model.Forward)
		attnOut, newState = gdn.deltaNetChunked(ctx, qConv, kConv, vConv, gate, beta, state, opts.masks, opts)
	}

	access.updateDelta(newState)

	// Apply gated normalization
	attnOut2D := attnOut.Contiguous(ctx, headVDim, numVHeads*nSeqTokens*nSeqs)
	z2D := z.Contiguous(ctx, headVDim, numVHeads*nSeqTokens*nSeqs)

	// norm(attnOut, z) = RMSNorm(attnOut) * silu(z)
	attnOutNorm := gdn.SSMNorm.Forward(ctx, attnOut2D, opts.eps)
	zSilu := z2D.SILU(ctx)
	attnOutGated := attnOutNorm.Mul(ctx, zSilu)

	// Reshape for output projection
	finalOutput := attnOutGated.Reshape(ctx, headVDim*numVHeads, nSeqTokens, nSeqs)

	out := gdn.SSMOut.Forward(ctx, finalOutput)
	return out.Reshape(ctx, out.Dim(0), nSeqTokens*nSeqs), nil
}

// deltaNetAutoregressive implements single-token state update.
// NOTE: Assumes headKDim == headVDim (state shape is [headVDim, headVDim, numVHeads, nSeqs]).
func (gdn *GatedDeltaNet) deltaNetAutoregressive(
	ctx ml.Context,
	q, k, v, gate, beta, state ml.Tensor,
	opts *Options,
) (ml.Tensor, ml.Tensor) {
	numVHeads := v.Dim(1)
	headVDim := v.Dim(0)
	nSeqs := q.Dim(3)

	// L2 normalize Q and K
	q = q.L2Norm(ctx, opts.eps)
	k = k.L2Norm(ctx, opts.eps)

	// Scale Q
	scale := 1.0 / math.Sqrt(float64(headVDim))
	q = q.Scale(ctx, scale)

	// Sigmoid beta
	beta = beta.Sigmoid(ctx)

	// Reshape state: [headVDim, headVDim, numVHeads, nSeqs]
	state = state.Reshape(ctx, headVDim, headVDim, numVHeads, nSeqs)

	// Reshape gate and beta for broadcasting
	gT := gate.Permute(ctx, 1, 0, 2, 3).Reshape(ctx, 1, 1, numVHeads, nSeqs)
	betaT := beta.Permute(ctx, 1, 0, 2, 3).Reshape(ctx, 1, 1, numVHeads, nSeqs)

	// Apply exponential to gate
	gT = gT.Exp(ctx)

	// state = state * g_t
	state = state.Mul(ctx, gT)

	// kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
	kTUnsqueezed := k.Reshape(ctx, 1, headVDim, numVHeads, nSeqs)
	kvMem := state.Mul(ctx, kTUnsqueezed)
	// Sum over dim=-2 (second dimension after permute)
	kvMem = kvMem.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	kvMem = kvMem.SumRows(ctx)
	kvMem = kvMem.Permute(ctx, 1, 0, 2, 3)

	// v_t with singleton dimension
	vT := v.Reshape(ctx, headVDim, 1, numVHeads, nSeqs)

	// delta = (v_t - kv_mem) * beta_t
	vDiff := vT.Sub(ctx, kvMem)
	delta := vDiff.Mul(ctx, betaT)

	// state = state + k_t.unsqueeze(-1) * delta
	kTUnsqueezedBroad := kTUnsqueezed.Repeat4D(ctx, headVDim, headVDim, numVHeads, nSeqs)
	kTDelta := kTUnsqueezedBroad.Mul(ctx, delta)
	state = state.Add(ctx, kTDelta)

	// core_attn_out = (state * q_t.unsqueeze(-1)).sum(dim=-2)
	qTUnsqueezed := q.Reshape(ctx, 1, headVDim, numVHeads, nSeqs)
	stateQ := state.Mul(ctx, qTUnsqueezed)
	stateQ = stateQ.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	coreAttnOut := stateQ.SumRows(ctx)
	coreAttnOut = coreAttnOut.Permute(ctx, 1, 0, 2, 3)

	newState := state.Reshape(ctx, headVDim, headVDim*numVHeads, nSeqs)
	return coreAttnOut.Reshape(ctx, headVDim, numVHeads, 1, nSeqs), newState
}

// deltaNetChunked implements chunked computation for prefill.
// NOTE: Assumes headKDim == headVDim (state shape is [headVDim, headVDim, numVHeads, nSeqs]).
func (gdn *GatedDeltaNet) deltaNetChunked(
	ctx ml.Context,
	q, k, v, gate, beta, state ml.Tensor,
	masks *Masks,
	opts *Options,
) (ml.Tensor, ml.Tensor) {
	headKDim := q.Dim(0)
	numVHeads := v.Dim(1)
	headVDim := v.Dim(0)
	nTokens := q.Dim(2)
	nSeqs := q.Dim(3)

	// L2 normalize Q and K
	q = q.L2Norm(ctx, opts.eps)
	k = k.L2Norm(ctx, opts.eps)

	// Scale Q
	scale := 1.0 / math.Sqrt(float64(headVDim))
	q = q.Scale(ctx, scale)

	// Sigmoid beta
	beta = beta.Sigmoid(ctx)

	// Permute tensors for chunked computation
	q = q.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, headKDim, nTokens, numVHeads, nSeqs)
	k = k.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, headKDim, nTokens, numVHeads, nSeqs)
	v = v.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, headVDim, nTokens, numVHeads, nSeqs)
	gate = gate.Permute(ctx, 2, 0, 3, 1).Contiguous(ctx, nTokens, 1, numVHeads, nSeqs)

	beta = beta.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)
	state = state.Reshape(ctx, headVDim, headVDim, numVHeads, nSeqs)

	// Compute padding
	pad := (chunkSize - nTokens%chunkSize) % chunkSize
	nChunks := (nTokens + pad) / chunkSize

	// Pad tensors
	if pad > 0 {
		q = q.Pad(ctx, 0, pad, 0, 0)
		k = k.Pad(ctx, 0, pad, 0, 0)
		v = v.Pad(ctx, 0, pad, 0, 0)
		gate = gate.Pad(ctx, pad, 0, 0, 0)
		beta = beta.Pad(ctx, 0, pad, 0, 0)
	}

	// Use pre-computed masks (passed in, not recreated)
	causalMask := masks.Causal
	identity := masks.Identity
	diagMask := masks.Diag
	identity4D := identity.Reshape(ctx, chunkSize, chunkSize, 1, 1)

	// v_beta = v * beta, k_beta = k * beta
	vBeta := v.Mul(ctx, beta)
	kBeta := k.Mul(ctx, beta)

	// Reshape for chunked computation
	q = q.Reshape(ctx, headKDim, chunkSize, nChunks, numVHeads*nSeqs)
	k = k.Reshape(ctx, headKDim, chunkSize, nChunks, numVHeads*nSeqs)
	kBeta = kBeta.Reshape(ctx, headKDim, chunkSize, nChunks, numVHeads*nSeqs)
	vBeta = vBeta.Reshape(ctx, headVDim, chunkSize, nChunks, numVHeads*nSeqs)

	gate = gate.Reshape(ctx, chunkSize, 1, nChunks, numVHeads*nSeqs)

	// g_cumsum = cumsum(gate)
	gCumsum := gate.CumSum(ctx)

	// Compute decay mask
	gcsI := gCumsum.Reshape(ctx, chunkSize, 1, nChunks, numVHeads*nSeqs)
	gcsJ := gCumsum.Reshape(ctx, 1, chunkSize, nChunks, numVHeads*nSeqs)
	gcsBroadcast := gcsJ.Repeat4D(ctx, chunkSize, chunkSize, nChunks, numVHeads*nSeqs)
	decayMask := gcsBroadcast.Sub(ctx, gcsI)

	decayMask = decayMask.Mul(ctx, diagMask)
	decayMask = decayMask.Exp(ctx)
	decayMask = decayMask.Mul(ctx, diagMask)

	// k @ k_beta^T
	kMulKBeta := k.Mulmat(ctx, kBeta)

	// k_decay = k @ k_beta^T * decay_mask
	kDecay := kMulKBeta.Mul(ctx, decayMask)

	// attn = -k_decay * causal_mask
	attn := kDecay.Neg(ctx).Mul(ctx, causalMask)

	// Triangular solve: (I - attn_lower)^-1 @ attn
	attnLower := attn.Mul(ctx, causalMask)
	lhs := attnLower.Neg(ctx).Add(ctx, identity4D)
	linSolve := lhs.SolveTri(ctx, attn, true, true, false)
	attn = linSolve.Mul(ctx, causalMask)
	attn = attn.Add(ctx, identity4D)

	// v = v_beta^T @ attn
	vBetaT := vBeta.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	v = vBetaT.Mulmat(ctx, attn)

	// Compute g_exp for state update
	gCumsumT := gCumsum.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	gExp := gCumsumT.Exp(ctx)

	// kbeta_gexp = k_beta * g_exp
	kBetaGExp := kBeta.Mul(ctx, gExp)

	// k_cumdecay = attn @ kbeta_gexp^T
	kBetaGExpT := kBetaGExp.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	kCumdecay := attn.Mulmat(ctx, kBetaGExpT)
	kCumdecay = kCumdecay.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	// Pre-compute attn_kq = (k @ q) * decay_mask * diag_mask
	attnKQ := k.Mulmat(ctx, q)
	attnKQ = attnKQ.Mul(ctx, decayMask)
	attnKQ = attnKQ.Mul(ctx, diagMask)

	// Pre-compute g_last and key_gdiff
	// g_last = view of last element in g_cumsum along chunk_size dimension
	// We need to get the last row of gCumsum: shape [chunkSize, 1, nChunks, H*n_seqs] -> [1, 1, nChunks, H*n_seqs]
	gLast := gCumsum.Slice(ctx, 0, chunkSize-1, chunkSize, 1).Contiguous(ctx, 1, 1, nChunks, numVHeads*nSeqs)
	gLastExp := gLast.Exp(ctx)

	// g_diff = -(g_cumsum - g_last) = g_last - g_cumsum
	gDiff := gCumsum.Neg(ctx).Add(ctx, gLast)
	gDiffExp := gDiff.Exp(ctx)

	// Reshapes g_diff_exp to [1, chunkSize, nChunks, ...]
	gDiffExpReshaped := gDiffExp.Reshape(ctx, 1, chunkSize, nChunks, numVHeads*nSeqs)
	keyGDiff := k.Mul(ctx, gDiffExpReshaped)
	keyGDiffT := keyGDiff.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	// Process chunks and update state
	var coreAttnOut ml.Tensor
	newState := state

	for chunk := range nChunks {
		qChunk := q.Slice(ctx, 2, chunk, chunk+1, 1)
		vChunk := v.Slice(ctx, 2, chunk, chunk+1, 1)
		gExpChunk := gExp.Slice(ctx, 2, chunk, chunk+1, 1)
		kCumdecayChunk := kCumdecay.Slice(ctx, 2, chunk, chunk+1, 1)
		attnChunk := attnKQ.Slice(ctx, 2, chunk, chunk+1, 1) // Pre-computed!

		// state^T - permute is needed but Contiguous creates a copy
		stateT := newState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx, headVDim, headVDim, 1, numVHeads*nSeqs)

		// v_prime = k_cumdecay @ state
		vPrime := stateT.Mulmat(ctx, kCumdecayChunk)

		// v_new = v - v_prime
		vNew := vChunk.Sub(ctx, vPrime)
		vNewT := vNew.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

		// attn_inter = (q * g_exp) @ state
		qGExp := qChunk.Mul(ctx, gExpChunk)
		attnInter := stateT.Mulmat(ctx, qGExp)

		// core_attn_out = attn_inter + attn @ v_new
		vAttn := vNewT.Mulmat(ctx, attnChunk)
		coreAttnOutChunk := attnInter.Add(ctx, vAttn)

		if coreAttnOut == nil {
			coreAttnOut = coreAttnOutChunk
		} else {
			coreAttnOut = coreAttnOut.Concat(ctx, coreAttnOutChunk, 1)
		}

		// Update state for next chunk
		gExpLastChunk := gLastExp.Slice(ctx, 2, chunk, chunk+1, 1)
		kGDiffChunkT := keyGDiffT.Slice(ctx, 2, chunk, chunk+1, 1)
		kgdMulVNew := vNewT.Mulmat(ctx, kGDiffChunkT)

		// state = state * g_last + kgdmulvnew
		gExpLastReshaped := gExpLastChunk.Contiguous(ctx).Reshape(ctx, 1, 1, numVHeads, nSeqs)
		newState = newState.Mul(ctx, gExpLastReshaped)
		newState = newState.Add(ctx, kgdMulVNew.Reshape(ctx, headVDim, headVDim, numVHeads, nSeqs))
	}

	// Final reshape
	coreAttnOut = coreAttnOut.Contiguous(ctx, headVDim, chunkSize*nChunks, numVHeads, nSeqs)

	// Slice to remove padding
	if pad > 0 {
		coreAttnOut = coreAttnOut.Slice(ctx, 1, 0, nTokens, 1)
	}

	newStateFlat := newState.Reshape(ctx, headVDim, headVDim*numVHeads, nSeqs)
	return coreAttnOut.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, headVDim, numVHeads, nTokens, nSeqs), newStateFlat
}
