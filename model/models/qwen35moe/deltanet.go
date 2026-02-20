package qwen35moe

import (
	"errors"
	"log/slog"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

const chunkSize = 64

const (
	TriTypeUpperDiag = 0
	TriTypeUpper     = 1
	TriTypeLowerDiag = 2
	TriTypeLower     = 3
)

type convKernel struct {
	Weight ml.Tensor `gguf:"weight"`
}

type Masks struct {
	Causal   ml.Tensor
	Identity ml.Tensor
	Diag     ml.Tensor
}

type GatedDeltaNet struct {
	SSMQKV       *nn.Linear  `gguf:"attn_qkv,alt:wqkv"`
	SSMQKVGate   *nn.Linear  `gguf:"attn_gate,alt:wqkv_gate"`
	SSMIn        *nn.Linear  `gguf:"ssm_in"`
	SSMBetaAlpha *nn.Linear  `gguf:"ssm_ba,alt:ssm_beta_alpha"`
	SSMConv1D    *convKernel `gguf:"ssm_conv1d"`
	SSMDT        ml.Tensor   `gguf:"ssm_dt"`
	SSMA         ml.Tensor   `gguf:"ssm_a"`
	SSMNorm      *nn.RMSNorm `gguf:"ssm_norm"`
	SSMOut       *nn.Linear  `gguf:"ssm_out"`

	Layer int
}

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

func (gdn *GatedDeltaNet) splitQKVZ(
	ctx ml.Context,
	hiddenStates ml.Tensor,
	headKDim, headVDim, numKHeads, numVHeads, nSeqTokens, nSeqs int,
) (ml.Tensor, ml.Tensor, error) {
	if numKHeads == 0 || numVHeads == 0 || headKDim == 0 || headVDim == 0 {
		return nil, nil, errors.New("qwen35moe: invalid linear attention dimensions for ssm_in split")
	}
	if numVHeads%numKHeads != 0 {
		return nil, nil, errors.New("qwen35moe: num_value_heads must be divisible by num_key_heads for ssm_in split")
	}
	vPerHead := headVDim * (numVHeads / numKHeads)
	qkvzDim := 2*headKDim + 2*vPerHead

	mixed := gdn.SSMIn.Forward(ctx, hiddenStates)
	mixed = mixed.Reshape(ctx, qkvzDim, numKHeads, nSeqTokens, nSeqs)

	qSlice := mixed.Slice(ctx, 0, 0, headKDim, 1)
	kSlice := mixed.Slice(ctx, 0, headKDim, 2*headKDim, 1)
	vSlice := mixed.Slice(ctx, 0, 2*headKDim, 2*headKDim+vPerHead, 1)
	zSlice := mixed.Slice(ctx, 0, 2*headKDim+vPerHead, 2*headKDim+2*vPerHead, 1)

	qFlat := qSlice.Contiguous(ctx, headKDim*numKHeads, nSeqTokens, nSeqs)
	kFlat := kSlice.Contiguous(ctx, headKDim*numKHeads, nSeqTokens, nSeqs)
	vFlat := vSlice.Contiguous(ctx, headVDim*numVHeads, nSeqTokens, nSeqs)
	zFlat := zSlice.Contiguous(ctx, headVDim*numVHeads, nSeqTokens, nSeqs)

	qkvMixed := qFlat.Concat(ctx, kFlat, 0)
	qkvMixed = qkvMixed.Concat(ctx, vFlat, 0)

	return qkvMixed, zFlat, nil
}

func (gdn *GatedDeltaNet) Forward(ctx ml.Context, hiddenStates, _ ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error) {
	layer := gdn.Layer
	nSeqTokens := hiddenStates.Dim(1)
	nSeqs := hiddenStates.Dim(2)
	if cache != nil && cache.IsSupportedForBatch() {
		seqTokens := cache.seqTokens()
		seqs := cache.numSeqs()
		if seqTokens > 0 && seqs > 0 {
			if nSeqs > 1 {
				if nSeqTokens != seqTokens || nSeqs != seqs {
					return nil, ErrUnsupportedBatchLayout
				}
			} else {
				if nSeqTokens != seqTokens*seqs {
					return nil, ErrUnsupportedBatchLayout
				}
				hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), seqTokens, seqs)
				nSeqTokens = seqTokens
				nSeqs = seqs
			}
		}
	}

	headKDim := opts.ssmDState
	numKHeads := opts.ssmNGroup
	numVHeads := opts.ssmDtRank
	headVDim := opts.ssmDInner / numVHeads
	convKernelSize := opts.convKernelSize

	if gdn.SSMBetaAlpha == nil {
		return nil, errors.New("qwen35moe: missing ssm_ba (beta/alpha projection)")
	}
	mixedBA := gdn.SSMBetaAlpha.Forward(ctx, hiddenStates)
	qkvDim := headKDim*numKHeads*2 + headVDim*numVHeads

	var (
		qkvMixed ml.Tensor
		z        ml.Tensor
	)
	switch {
	case gdn.SSMQKV != nil && gdn.SSMQKVGate != nil:
		qkvMixed = gdn.SSMQKV.Forward(ctx, hiddenStates).Reshape(ctx, qkvDim, nSeqTokens, nSeqs)
		z = gdn.SSMQKVGate.Forward(ctx, hiddenStates)
	case gdn.SSMIn != nil:
		var err error
		qkvMixed, z, err = gdn.splitQKVZ(ctx, hiddenStates, headKDim, headVDim, numKHeads, numVHeads, nSeqTokens, nSeqs)
		if err != nil {
			return nil, err
		}
	default:
		return nil, errors.New("qwen35moe: missing attn_qkv/attn_gate or legacy ssm_in projections")
	}

	baNewDim := 2 * numVHeads / numKHeads
	mixedBAReshaped := mixedBA.Reshape(ctx, baNewDim, numKHeads, nSeqTokens, nSeqs)

	betaSize := numVHeads / numKHeads
	alphaSize := numVHeads / numKHeads

	b := mixedBAReshaped.Slice(ctx, 0, 0, betaSize, 1)
	a := mixedBAReshaped.Slice(ctx, 0, betaSize, betaSize+alphaSize, 1)

	beta := b.Contiguous(ctx, numVHeads, 1, nSeqTokens, nSeqs)
	alpha := a.Contiguous(ctx, numVHeads, nSeqTokens, nSeqs)

	alphaBiased := alpha.Add(ctx, gdn.SSMDT)
	alphaSoftplus := alphaBiased.Softplus(ctx)
	gate := alphaSoftplus.Mul(ctx, gdn.SSMA)
	qkvMixed = qkvMixed.Permute(ctx, 1, 0, 2, 3)

	convStates, err := cache.ConvState(ctx, layer)
	if err != nil {
		slog.Warn("qwen35moe: failed to get conv state, using zeros", "layer", layer, "error", err)
		convStates = ctx.Input().Zeros(ml.DTypeF32, convKernelSize-1, qkvDim, nSeqs)
	}

	convStates = convStates.Reshape(ctx, convKernelSize-1, qkvDim, nSeqs)

	convInput := convStates.Concat(ctx, qkvMixed, 0)

	lastConvStates := convInput.Slice(ctx, 0, nSeqTokens, nSeqTokens+convKernelSize-1, 1)
	cache.UpdateConvState(ctx, layer, lastConvStates)

	convOutput := convInput.SSMConv(ctx, gdn.SSMConv1D.Weight)
	convOutput = convOutput.SILU(ctx)

	convQKVMix := convOutput.Contiguous(ctx, qkvDim, nSeqTokens*nSeqs)

	qConv := convQKVMix.Slice(ctx, 0, 0, headKDim*numKHeads, 1)
	kConv := convQKVMix.Slice(ctx, 0, headKDim*numKHeads, 2*headKDim*numKHeads, 1)
	vConv := convQKVMix.Slice(ctx, 0, 2*headKDim*numKHeads, qkvDim, 1)

	qConv = qConv.Contiguous(ctx, headKDim, numKHeads, nSeqTokens, nSeqs)
	kConv = kConv.Contiguous(ctx, headKDim, numKHeads, nSeqTokens, nSeqs)
	vConv = vConv.Contiguous(ctx, headVDim, numVHeads, nSeqTokens, nSeqs)

	state, err := cache.DeltaState(ctx, layer, headVDim, numVHeads)
	if err != nil {
		slog.Warn("qwen35moe: failed to get delta state, using zeros", "layer", layer, "error", err)
		state = ctx.Input().Zeros(ml.DTypeF32, headVDim, headVDim*numVHeads, nSeqs)
	}
	state = state.Reshape(ctx, headVDim, headVDim*numVHeads, 1, nSeqs)

	if numKHeads != numVHeads {
		repeatFactor := numVHeads / numKHeads

		qReshaped := qConv.Reshape(ctx, headKDim, 1, numKHeads*nSeqTokens*nSeqs)
		kReshaped := kConv.Reshape(ctx, headKDim, 1, numKHeads*nSeqTokens*nSeqs)

		qRepeated := qReshaped.Repeat4D(ctx, headKDim, repeatFactor, numKHeads*nSeqTokens*nSeqs, 1)
		kRepeated := kReshaped.Repeat4D(ctx, headKDim, repeatFactor, numKHeads*nSeqTokens*nSeqs, 1)

		qConv = qRepeated.Reshape(ctx, headKDim, numKHeads*repeatFactor, nSeqTokens, nSeqs)
		kConv = kRepeated.Reshape(ctx, headKDim, numKHeads*repeatFactor, nSeqTokens, nSeqs)
	}

	var attnOut ml.Tensor
	if nSeqTokens == 1 {
		attnOut = gdn.deltaNetAutoregressive(ctx, qConv, kConv, vConv, gate, beta, state, opts, layer, cache)
	} else {
		attnOut = gdn.deltaNetChunked(ctx, qConv, kConv, vConv, gate, beta, state, opts.masks, opts, layer, cache)
	}

	attnOut2D := attnOut.Contiguous(ctx, headVDim, numVHeads*nSeqTokens*nSeqs)
	z2D := z.Contiguous(ctx, headVDim, numVHeads*nSeqTokens*nSeqs)

	attnOutNorm := gdn.SSMNorm.Forward(ctx, attnOut2D, opts.eps)
	zSilu := z2D.SILU(ctx)
	attnOutGated := attnOutNorm.Mul(ctx, zSilu)

	finalOutput := attnOutGated.Reshape(ctx, headVDim*numVHeads, nSeqTokens, nSeqs)

	out := gdn.SSMOut.Forward(ctx, finalOutput)
	return out.Reshape(ctx, out.Dim(0), nSeqTokens*nSeqs), nil
}

func (gdn *GatedDeltaNet) deltaNetAutoregressive(
	ctx ml.Context,
	q, k, v, gate, beta, state ml.Tensor,
	opts *Options,
	layer int,
	cache *HybridCache,
) ml.Tensor {
	numVHeads := v.Dim(1)
	headVDim := v.Dim(0)
	nSeqs := q.Dim(3)

	q = q.L2Norm(ctx, opts.eps)
	k = k.L2Norm(ctx, opts.eps)

	scale := 1.0 / math.Sqrt(float64(headVDim))
	q = q.Scale(ctx, scale)

	beta = beta.Sigmoid(ctx)

	state = state.Reshape(ctx, headVDim, headVDim, numVHeads, nSeqs)

	gT := gate.Permute(ctx, 1, 0, 2, 3).Reshape(ctx, 1, 1, numVHeads, nSeqs)
	betaT := beta.Permute(ctx, 1, 0, 2, 3).Reshape(ctx, 1, 1, numVHeads, nSeqs)

	gT = gT.Exp(ctx)

	state = state.Mul(ctx, gT)

	kTUnsqueezed := k.Reshape(ctx, 1, headVDim, numVHeads, nSeqs)
	kvMem := state.Mul(ctx, kTUnsqueezed)
	kvMem = kvMem.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	kvMem = kvMem.SumRows(ctx)
	kvMem = kvMem.Permute(ctx, 1, 0, 2, 3)

	vT := v.Reshape(ctx, headVDim, 1, numVHeads, nSeqs)

	vDiff := vT.Sub(ctx, kvMem)
	delta := vDiff.Mul(ctx, betaT)

	kTUnsqueezedBroad := kTUnsqueezed.Repeat4D(ctx, headVDim, headVDim, numVHeads, nSeqs)
	kTDelta := kTUnsqueezedBroad.Mul(ctx, delta)
	state = state.Add(ctx, kTDelta)

	qTUnsqueezed := q.Reshape(ctx, 1, headVDim, numVHeads, nSeqs)
	stateQ := state.Mul(ctx, qTUnsqueezed)
	stateQ = stateQ.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	coreAttnOut := stateQ.SumRows(ctx)
	coreAttnOut = coreAttnOut.Permute(ctx, 1, 0, 2, 3)

	cache.UpdateDeltaState(ctx, layer, state.Reshape(ctx, headVDim, headVDim*numVHeads, nSeqs))

	return coreAttnOut.Reshape(ctx, headVDim, numVHeads, 1, nSeqs)
}

func (gdn *GatedDeltaNet) deltaNetChunked(
	ctx ml.Context,
	q, k, v, gate, beta, state ml.Tensor,
	masks *Masks,
	opts *Options,
	layer int,
	cache *HybridCache,
) ml.Tensor {
	headKDim := q.Dim(0)
	numVHeads := v.Dim(1)
	headVDim := v.Dim(0)
	nTokens := q.Dim(2)
	nSeqs := q.Dim(3)

	q = q.L2Norm(ctx, opts.eps)
	k = k.L2Norm(ctx, opts.eps)

	scale := 1.0 / math.Sqrt(float64(headVDim))
	q = q.Scale(ctx, scale)

	beta = beta.Sigmoid(ctx)

	q = q.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, headKDim, nTokens, numVHeads, nSeqs)
	k = k.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, headKDim, nTokens, numVHeads, nSeqs)
	v = v.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, headVDim, nTokens, numVHeads, nSeqs)
	gate = gate.Permute(ctx, 2, 0, 3, 1).Contiguous(ctx, nTokens, 1, numVHeads, nSeqs)

	beta = beta.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)
	state = state.Reshape(ctx, headVDim, headVDim, numVHeads, nSeqs)

	pad := (chunkSize - nTokens%chunkSize) % chunkSize
	nChunks := (nTokens + pad) / chunkSize

	if pad > 0 {
		q = q.Pad(ctx, 0, pad, 0, 0)
		k = k.Pad(ctx, 0, pad, 0, 0)
		v = v.Pad(ctx, 0, pad, 0, 0)
		gate = gate.Pad(ctx, pad, 0, 0, 0)
		beta = beta.Pad(ctx, 0, pad, 0, 0)
	}

	causalMask := masks.Causal
	identity := masks.Identity
	diagMask := masks.Diag
	identity4D := identity.Reshape(ctx, chunkSize, chunkSize, 1, 1)

	vBeta := v.Mul(ctx, beta)
	kBeta := k.Mul(ctx, beta)

	q = q.Reshape(ctx, headKDim, chunkSize, nChunks, numVHeads*nSeqs)
	k = k.Reshape(ctx, headKDim, chunkSize, nChunks, numVHeads*nSeqs)
	kBeta = kBeta.Reshape(ctx, headKDim, chunkSize, nChunks, numVHeads*nSeqs)
	vBeta = vBeta.Reshape(ctx, headVDim, chunkSize, nChunks, numVHeads*nSeqs)

	gate = gate.Reshape(ctx, chunkSize, 1, nChunks, numVHeads*nSeqs)

	gCumsum := gate.CumSum(ctx)

	gcsI := gCumsum.Reshape(ctx, chunkSize, 1, nChunks, numVHeads*nSeqs)
	gcsJ := gCumsum.Reshape(ctx, 1, chunkSize, nChunks, numVHeads*nSeqs)
	gcsBroadcast := gcsJ.Repeat4D(ctx, chunkSize, chunkSize, nChunks, numVHeads*nSeqs)
	decayMask := gcsBroadcast.Sub(ctx, gcsI)

	decayMask = decayMask.Mul(ctx, diagMask)
	decayMask = decayMask.Exp(ctx)
	decayMask = decayMask.Mul(ctx, diagMask)

	kMulKBeta := k.Mulmat(ctx, kBeta)

	kDecay := kMulKBeta.Mul(ctx, decayMask)

	attn := kDecay.Neg(ctx).Mul(ctx, causalMask)

	attnLower := attn.Mul(ctx, causalMask)
	lhs := attnLower.Neg(ctx).Add(ctx, identity4D)
	linSolve := lhs.SolveTri(ctx, attn, true, true, false)
	attn = linSolve.Mul(ctx, causalMask)
	attn = attn.Add(ctx, identity4D)

	vBetaT := vBeta.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	v = vBetaT.Mulmat(ctx, attn)

	gCumsumT := gCumsum.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	gExp := gCumsumT.Exp(ctx)

	kBetaGExp := kBeta.Mul(ctx, gExp)

	kBetaGExpT := kBetaGExp.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	kCumdecay := attn.Mulmat(ctx, kBetaGExpT)
	kCumdecay = kCumdecay.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	attnKQ := k.Mulmat(ctx, q)
	attnKQ = attnKQ.Mul(ctx, decayMask)
	attnKQ = attnKQ.Mul(ctx, diagMask)

	gLast := gCumsum.Slice(ctx, 0, chunkSize-1, chunkSize, 1).Contiguous(ctx, 1, 1, nChunks, numVHeads*nSeqs)
	gLastExp := gLast.Exp(ctx)

	gDiff := gCumsum.Neg(ctx).Add(ctx, gLast)
	gDiffExp := gDiff.Exp(ctx)

	gDiffExpReshaped := gDiffExp.Reshape(ctx, 1, chunkSize, nChunks, numVHeads*nSeqs)
	keyGDiff := k.Mul(ctx, gDiffExpReshaped)
	keyGDiffT := keyGDiff.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	var coreAttnOut ml.Tensor
	newState := state

	for chunk := range nChunks {
		qChunk := q.Slice(ctx, 2, chunk, chunk+1, 1)
		vChunk := v.Slice(ctx, 2, chunk, chunk+1, 1)
		gExpChunk := gExp.Slice(ctx, 2, chunk, chunk+1, 1)
		kCumdecayChunk := kCumdecay.Slice(ctx, 2, chunk, chunk+1, 1)
		attnChunk := attnKQ.Slice(ctx, 2, chunk, chunk+1, 1)

		stateT := newState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx, headVDim, headVDim, 1, numVHeads*nSeqs)

		vPrime := stateT.Mulmat(ctx, kCumdecayChunk)

		vNew := vChunk.Sub(ctx, vPrime)
		vNewT := vNew.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

		qGExp := qChunk.Mul(ctx, gExpChunk)
		attnInter := stateT.Mulmat(ctx, qGExp)

		vAttn := vNewT.Mulmat(ctx, attnChunk)
		coreAttnOutChunk := attnInter.Add(ctx, vAttn)

		if coreAttnOut == nil {
			coreAttnOut = coreAttnOutChunk
		} else {
			coreAttnOut = coreAttnOut.Concat(ctx, coreAttnOutChunk, 1)
		}

		gExpLastChunk := gLastExp.Slice(ctx, 2, chunk, chunk+1, 1)
		kGDiffChunkT := keyGDiffT.Slice(ctx, 2, chunk, chunk+1, 1)
		kgdMulVNew := vNewT.Mulmat(ctx, kGDiffChunkT)

		gExpLastReshaped := gExpLastChunk.Contiguous(ctx).Reshape(ctx, 1, 1, numVHeads, nSeqs)
		newState = newState.Mul(ctx, gExpLastReshaped)
		newState = newState.Add(ctx, kgdMulVNew.Reshape(ctx, headVDim, headVDim, numVHeads, nSeqs))
	}

	coreAttnOut = coreAttnOut.Contiguous(ctx, headVDim, chunkSize*nChunks, numVHeads, nSeqs)

	if pad > 0 {
		coreAttnOut = coreAttnOut.Slice(ctx, 1, 0, nTokens, 1)
	}

	cache.UpdateDeltaState(ctx, layer, newState.Reshape(ctx, headVDim, headVDim*numVHeads, nSeqs))

	return coreAttnOut.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, headVDim, numVHeads, nTokens, nSeqs)
}
