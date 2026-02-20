package qwen35

import (
	"errors"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

var ErrUnsupportedBatchLayout = errors.New("qwen35: unsupported batch layout")

type FullAttention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *FullAttention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error) {
	hiddenDim := hiddenStates.Dim(0)
	batchSize := hiddenStates.Dim(1)
	nSeqs := hiddenStates.Dim(2) // 0 if 2D tensor

	if cache != nil && cache.IsSupportedForBatch() {
		seqTokens := cache.seqTokens()
		seqs := cache.numSeqs()
		if seqTokens > 0 && seqs > 0 {
			if nSeqs > 0 {
				// 3D tensor: [hiddenDim, seqTokens, nSeqs]
				if batchSize != seqTokens || nSeqs != seqs {
					return nil, ErrUnsupportedBatchLayout
				}
				hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, seqTokens*seqs)
				batchSize = seqTokens * seqs
			} else if batchSize != seqTokens*seqs {
				return nil, ErrUnsupportedBatchLayout
			}
		}
	}
	headDim := opts.headDim()
	numHeads := opts.numHeads

	qFull := sa.Query.Forward(ctx, hiddenStates)

	qFull = qFull.Reshape(ctx, headDim*2, numHeads, batchSize)

	query := qFull.Slice(ctx, 0, 0, headDim, 1)
	gate := qFull.Slice(ctx, 0, headDim, headDim*2, 1)

	query = query.Contiguous(ctx, headDim, numHeads, batchSize)

	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	numKVHeads := key.Dim(0) / headDim

	key = key.Reshape(ctx, headDim, numKVHeads, batchSize)
	value = value.Reshape(ctx, headDim, numKVHeads, batchSize)

	query = sa.QueryNorm.Forward(ctx, query, opts.eps)
	key = sa.KeyNorm.Forward(ctx, key, opts.eps)

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	scale := opts.attentionScale
	if scale == 0 {
		scale = 1.0 / math.Sqrt(float64(headDim))
	}
	attention := nn.Attention(ctx, query, key, value, scale, cache)

	attention = attention.Reshape(ctx, headDim*numHeads, batchSize)

	gate = gate.Contiguous(ctx, headDim*numHeads, batchSize)
	gateSigmoid := gate.Sigmoid(ctx)
	attention = attention.Mul(ctx, gateSigmoid)

	return sa.Output.Forward(ctx, attention), nil
}
