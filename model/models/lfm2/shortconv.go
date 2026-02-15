package lfm2

import (
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type shortConvKernel struct {
	Weight ml.Tensor `gguf:"weight"`
}

// ShortConv implements the LFM2 short-convolution block (GGML_OP_SSM_CONV) with a recurrent
// state stored in the HybridCache.
type ShortConv struct {
	Conv    *shortConvKernel `gguf:"shortconv.conv"`
	InProj  *nn.Linear       `gguf:"shortconv.in_proj"`
	OutProj *nn.Linear       `gguf:"shortconv.out_proj"`
}

func (sc *ShortConv) Forward(ctx ml.Context, hiddenStates ml.Tensor, _ ml.Tensor, cache *HybridCache, layer int, opts *Options) ml.Tensor {
	nSeqs := cache.numSeqs()
	seqTokens := cache.seqTokens()
	hiddenSize := hiddenStates.Dim(0)
	if nSeqs <= 0 || seqTokens <= 0 || hiddenStates.Dim(1) != nSeqs*seqTokens {
		panic("lfm2: unsupported batch layout for shortconv")
	}

	bcx := sc.InProj.Forward(ctx, hiddenStates).Reshape(ctx, 3*hiddenSize, seqTokens, nSeqs)

	elementSize := bcx.Stride(0)
	b := bcx.View(ctx, 0*hiddenSize*elementSize, hiddenSize, bcx.Stride(1), seqTokens, bcx.Stride(2), nSeqs)
	c := bcx.View(ctx, 1*hiddenSize*elementSize, hiddenSize, bcx.Stride(1), seqTokens, bcx.Stride(2), nSeqs)
	x := bcx.View(ctx, 2*hiddenSize*elementSize, hiddenSize, bcx.Stride(1), seqTokens, bcx.Stride(2), nSeqs)

	bx := b.Mul(ctx, x).Permute(ctx, 1, 0, 2, 3)

	state, err := cache.ConvState(ctx, layer)
	if err != nil {
		panic("lfm2: failed to get conv state: " + err.Error())
	}
	sx := state.Concat(ctx, bx, 0)

	convOut := sx.SSMConv(ctx, sc.Conv.Weight)
	y := c.Mul(ctx, convOut)

	dConv := sx.Dim(0) - seqTokens
	cache.UpdateConvState(ctx, layer, sx.Slice(ctx, 0, sx.Dim(0)-dConv, sx.Dim(0), 1))

	return sc.OutProj.Forward(ctx, y.Reshape(ctx, hiddenSize, seqTokens*nSeqs))
}
