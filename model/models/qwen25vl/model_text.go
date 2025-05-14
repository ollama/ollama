package qwen25vl

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model/input"
)

type TextOptions struct {
	ctxLen, hiddenSize, numHeads, numKVHeads int
	eps, ropeBase, ropeScale                 float32
	ropeDim, defaultContextLen               uint32
}

type TextModel struct {
	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*TextOptions
}

func NewTextModel(c fs.Config) *TextModel {
	m := TextModel{
		Layers: make([]Layer, c.Uint("block_count")),
		TextOptions: &TextOptions{
			ctxLen:            int(c.Uint("context_length")),
			hiddenSize:        int(c.Uint("embedding_length")),
			numHeads:          int(c.Uint("attention.head_count")),
			numKVHeads:        int(c.Uint("attention.head_count_kv")),
			eps:               c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:          c.Float("rope.freq_base"),
			ropeScale:         c.Float("rope.freq_scale", 1),
			ropeDim:           c.Uint("rope.dimension_count", 128),
			defaultContextLen: c.Uint("context_length", 128000),
		},
	}

	return &m
}

// SelfAttention implements the multi-head self-attention mechanism
// with separate projections for query, key, value and output transformations
type SelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, headDim, opts.numHeads, batchSize)
	q = q.RoPE(ctx, positionIDs, nil, opts.ropeDim, 2, opts.ropeBase, opts.ropeScale, ml.WithContextLen(opts.defaultContextLen))

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	k = k.RoPE(ctx, positionIDs, nil, opts.ropeDim, 2, opts.ropeBase, opts.ropeScale, ml.WithContextLen(opts.defaultContextLen))

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	scaleFactor := 1.0 / math.Sqrt(float64(headDim))
	kqv := nn.Attention(ctx, q, k, v, scaleFactor, cache)
	kqv = kqv.Reshape(ctx, opts.hiddenSize, batchSize)

	return sa.Output.Forward(ctx, kqv)
}

// Shift applies rotary position embeddings to the key tensor for causal attention caching
func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return key.RoPE(ctx, shift, nil, m.ropeDim, 2, m.ropeBase, m.ropeScale, ml.WithContextLen(m.defaultContextLen)), nil
}

// MLP implements the feed-forward network component with SwiGLU activation
type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextOptions) ml.Tensor {
	// Apply SwiGLU activation gating
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenState))
	// Project back to hidden dimension
	return mlp.Down.Forward(ctx, hiddenState)
}

// Layer represents a single transformer layer combining self-attention and feed-forward components
type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positionIDs, outputs ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	// Self-attention branch with residual connection
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, positionIDs, cache, opts)

	// In the final layer (outputs != nil), optimize by pruning to just the token positions
	// we need logits for.
	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenState = hiddenState.Add(ctx, residual)
	// Feed-forward branch with residual connection
	residual = hiddenState
	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	return hiddenState.Add(ctx, residual)
}

func (m *TextModel) Forward(ctx ml.Context, inputs, positions, outputs ml.Tensor, batch input.Batch, cache kvcache.Cache) (ml.Tensor, error) {
	// Initial token embedding
	hiddenStates := m.TokenEmbedding.Forward(ctx, inputs).Duplicate(ctx)

	for _, mi := range batch.Multimodal {
		f32s := mi.Multimodal.(*chunk).floats()
		img, err := ctx.Input().FromFloatSlice(f32s, len(f32s)/m.hiddenSize, m.hiddenSize)
		if err != nil {
			panic(err)
		}

		ctx.Forward(img.Copy(ctx, hiddenStates.View(ctx, mi.Index*hiddenStates.Stride(1), img.Dim(0)*img.Dim(1))))
	}

	// Process through transformer layers
	for i, layer := range m.Layers {
		cache.SetLayer(i)

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, lastLayerOutputs, cache, m.TextOptions)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}
