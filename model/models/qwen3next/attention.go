package qwen3next

import (
	"errors"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// ErrUnsupportedBatchLayout is returned when the batch layout is incompatible
// with the attention layer requirements.
var ErrUnsupportedBatchLayout = errors.New("qwen3next: unsupported batch layout")

// FullAttention implements gated attention with QK normalization and sigmoid-gated output.
// Key differences from standard attention:
// - Q projection outputs 2x size (Q + gate interleaved)
// - Both Q and K have RMSNorm
// - Output is gated: attn * sigmoid(gate)
type FullAttention struct {
	Query     *nn.Linear  `gguf:"attn_q"` // outputs [n_embd_head * 2, n_head]
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *FullAttention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error) {
	// Use Dim() instead of Shape() for consistent behavior during graph construction
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

	// Q projection outputs query + gate interleaved
	qFull := sa.Query.Forward(ctx, hiddenStates)

	// Reshape to [headDim * 2, numHeads, batchSize]
	qFull = qFull.Reshape(ctx, headDim*2, numHeads, batchSize)

	// Split Q and gate along dimension 0
	// Q: first headDim elements, gate: second headDim elements
	query := qFull.Slice(ctx, 0, 0, headDim, 1)
	gate := qFull.Slice(ctx, 0, headDim, headDim*2, 1)

	// Make query contiguous for further operations
	query = query.Contiguous(ctx, headDim, numHeads, batchSize)

	// K and V projections
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	// Derive numKVHeads from tensor dimensions (per-layer value)
	numKVHeads := key.Dim(0) / headDim

	key = key.Reshape(ctx, headDim, numKVHeads, batchSize)
	value = value.Reshape(ctx, headDim, numKVHeads, batchSize)

	// Apply QK normalization
	query = sa.QueryNorm.Forward(ctx, query, opts.eps)
	key = sa.KeyNorm.Forward(ctx, key, opts.eps)

	// Apply RoPE
	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	// Standard attention computation
	scale := opts.attentionScale
	if scale == 0 {
		scale = 1.0 / math.Sqrt(float64(headDim))
	}
	attention := nn.Attention(ctx, query, key, value, scale, cache)

	// Flatten heads
	attention = attention.Reshape(ctx, headDim*numHeads, batchSize)

	// Apply sigmoid gate
	// gate shape: [headDim, numHeads, batchSize] -> [headDim*numHeads, batchSize]
	gate = gate.Contiguous(ctx, headDim*numHeads, batchSize)
	gateSigmoid := gate.Sigmoid(ctx)
	attention = attention.Mul(ctx, gateSigmoid)

	return sa.Output.Forward(ctx, attention), nil
}
