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

	// in_proj(x) -> split into B, C, X
	bcx := sc.InProj.Forward(ctx, hiddenStates).Reshape(ctx, 3*hiddenSize, seqTokens, nSeqs)
	bcx = bcx.Contiguous(ctx)
	bcxChunks := bcx.ChunkSections(ctx, 0, hiddenSize, hiddenSize, hiddenSize)
	b, c, x := bcxChunks[0], bcxChunks[1], bcxChunks[2]

	// bx := B * X (permute into [time, hidden, seq] for GGML_OP_SSM_CONV)
	// Always provide 4 dims to Permute: GGML frequently represents tensors as 4D views.
	bx := b.Mul(ctx, x).Permute(ctx, 1, 0, 2, 3)

	// sx := [state, bx] where state holds the last (L_cache-1) steps
	state := cache.ConvState(ctx, layer)
	sx := state.Concat(ctx, bx, 0)

	convOut := sx.SSMConv(ctx, sc.Conv.Weight)
	y := c.Mul(ctx, convOut)

	// Persist the new recurrent state (the last dConv steps of sx).
	dConv := sx.Dim(0) - seqTokens
	cache.UpdateConvState(ctx, layer, sx.Slice(ctx, 0, sx.Dim(0)-dConv, sx.Dim(0), 1))

	return sc.OutProj.Forward(ctx, y.Reshape(ctx, hiddenSize, seqTokens*nSeqs))
}
