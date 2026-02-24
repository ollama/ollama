package nemotronh

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// Attention implements simple attention without RoPE for Nemotron-H.
// Unlike Qwen3Next, Nemotron-H attention has:
// - No RoPE (position info comes from Mamba2 layers)
// - Standard scaled dot-product attention
type Attention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (a *Attention) Forward(ctx ml.Context, hiddenStates ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error) {
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
	batchSize := nSeqTokens
	hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, batchSize)

	headDim := opts.getHeadDim()
	if headDim <= 0 {
		return nil, fmt.Errorf("nemotronh: invalid attention head dimension %d", headDim)
	}

	// Q projection
	query := a.Query.Forward(ctx, hiddenStates)
	if query.Dim(0)%headDim != 0 {
		return nil, fmt.Errorf("nemotronh: query dim %d not divisible by head dim %d", query.Dim(0), headDim)
	}
	numHeads := query.Dim(0) / headDim
	query = query.Reshape(ctx, headDim, numHeads, batchSize)

	// K projection
	key := a.Key.Forward(ctx, hiddenStates)
	if key.Dim(0)%headDim != 0 {
		return nil, fmt.Errorf("nemotronh: key dim %d not divisible by head dim %d", key.Dim(0), headDim)
	}
	numKVHeads := key.Dim(0) / headDim
	key = key.Reshape(ctx, headDim, numKVHeads, batchSize)

	// V projection
	value := a.Value.Forward(ctx, hiddenStates)
	if value.Dim(0)%headDim != 0 {
		return nil, fmt.Errorf("nemotronh: value dim %d not divisible by head dim %d", value.Dim(0), headDim)
	}
	if value.Dim(0)/headDim != numKVHeads {
		return nil, fmt.Errorf("nemotronh: key heads %d and value heads %d do not match", numKVHeads, value.Dim(0)/headDim)
	}
	value = value.Reshape(ctx, headDim, numKVHeads, batchSize)

	// Standard attention computation (no RoPE)
	scale := opts.attentionScale
	if scale == 0 {
		scale = 1.0 / math.Sqrt(float64(headDim))
	}
	attention := nn.Attention(ctx, query, key, value, scale, cache)

	// Flatten heads
	attention = attention.Reshape(ctx, headDim*numHeads, batchSize)

	// Output projection
	return a.Output.Forward(ctx, attention), nil
}
