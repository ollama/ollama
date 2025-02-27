package mllama

import (
	"math"
	"slices"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type TextSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *TextSelfAttention) Forward(ctx ml.Context, hiddenState, positions, _ ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads

	query := sa.Query.Forward(ctx, hiddenState)
	query = query.Reshape(ctx, headDim, opts.numHeads, batchSize)
	query = query.RoPE(ctx, positions, opts.RopeFactors, opts.ropeDim, opts.ropeBase, opts.ropeScale)

	key := sa.Key.Forward(ctx, hiddenState)
	key = key.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	key = key.RoPE(ctx, positions, opts.RopeFactors, opts.ropeDim, opts.ropeBase, opts.ropeScale)

	value := sa.Value.Forward(ctx, hiddenState)
	value = value.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	cache.Put(ctx, key, value)
	key, value, mask := cache.Get(ctx)

	query = query.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	key = key.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	scaleFactor := 1.0 / math.Sqrt(float64(headDim))
	attention := nn.Attention(ctx, query, key, value, mask, scaleFactor)
	attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)

	return sa.Output.Forward(ctx, attention)
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	// This will only get called for layers in the cache, which are just the self attention layers
	return key.RoPE(ctx, shift, m.RopeFactors, m.ropeDim, m.ropeBase, m.ropeScale), nil
}

type TextMLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextModelOptions) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type TextSelfAttentionDecoderLayer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *TextSelfAttention

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP     *TextMLP
}

func (d *TextSelfAttentionDecoderLayer) Forward(ctx ml.Context, hiddenState, positions, outputs, mask, _, _ ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	residual := hiddenState

	hiddenState = d.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = d.SelfAttention.Forward(ctx, hiddenState, positions, mask, cache, opts)

	// In the final layer (outputs != nil), optimize by pruning to just the token positions
	// we need logits for.
	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = d.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = d.MLP.Forward(ctx, hiddenState, opts)
	return hiddenState.Add(ctx, residual)
}

type TextCrossAttention struct {
	QueryNorm *nn.RMSNorm `gguf:"cross_attn_q_norm"`
	Query     *nn.Linear  `gguf:"cross_attn_q_proj"`
	KeyNorm   *nn.RMSNorm `gguf:"cross_attn_k_norm"`
	Key       *nn.Linear  `gguf:"cross_attn_k_proj"`
	Value     *nn.Linear  `gguf:"cross_attn_v_proj"`
	Output    *nn.Linear  `gguf:"cross_attn_o_proj"`
}

func (ca *TextCrossAttention) Forward(ctx ml.Context, hiddenState, crossAttentionStates ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads

	query := ca.Query.Forward(ctx, hiddenState)
	query = query.Reshape(ctx, headDim, opts.numHeads, batchSize)
	query = ca.QueryNorm.Forward(ctx, query, opts.eps)

	var key, value, mask ml.Tensor
	if crossAttentionStates != nil {
		numVisionTokens, numTiles := crossAttentionStates.Dim(1), crossAttentionStates.Dim(2)

		key = ca.Key.Forward(ctx, crossAttentionStates)
		key = key.Reshape(ctx, headDim, opts.numKVHeads, numVisionTokens*numTiles)
		key = ca.KeyNorm.Forward(ctx, key, opts.eps)

		value = ca.Value.Forward(ctx, crossAttentionStates)
		value = value.Reshape(ctx, headDim, opts.numKVHeads, numVisionTokens*numTiles)

		cache.Put(ctx, key, value)
	} else {
		key, value, mask = cache.Get(ctx)
	}

	query = query.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	key = key.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	scaleFactor := 1.0 / math.Sqrt(float64(headDim))
	attention := nn.Attention(ctx, query, key, value, mask, scaleFactor)
	attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)

	return ca.Output.Forward(ctx, attention)
}

type TextCrossAttentionDecoderLayer struct {
	AttentionNorm  *nn.RMSNorm `gguf:"attn_norm"`
	CrossAttention *TextCrossAttention
	AttentionGate  ml.Tensor `gguf:"cross_attn_attn_gate"`

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP     *TextMLP
	MLPGate ml.Tensor `gguf:"cross_attn_mlp_gate"`
}

func (d *TextCrossAttentionDecoderLayer) Forward(ctx ml.Context, hiddenState, _, _, _, crossAttentionStates, crossAttentionMask ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	residual := hiddenState

	hiddenState = d.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = d.CrossAttention.Forward(ctx, hiddenState, crossAttentionStates, cache, opts)
	hiddenState = hiddenState.Mul(ctx, d.AttentionGate.Tanh(ctx))
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = d.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = d.MLP.Forward(ctx, hiddenState, opts)
	hiddenState = hiddenState.Mul(ctx, d.MLPGate.Tanh(ctx))
	return hiddenState.Add(ctx, residual)
}

type TextDecoderLayer interface {
	Forward(ctx ml.Context, hiddenState, positionIDs, outputs, mask, crossAttentionStates, crossAttentionMask ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor
}

type TextDecoder struct {
	Layers []TextDecoderLayer
}

func (d *TextDecoder) Forward(ctx ml.Context, hiddenState, positionIDs, outputs, mask, crossAttentionStates, crossAttentionMask ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	for i, layer := range d.Layers {
		layerType := selfAttentionLayer
		if slices.Contains(opts.crossAttentionLayers, uint32(i)) {
			layerType = crossAttentionLayer
		}

		cache.SetLayer(i)
		cache.SetLayerType(layerType)

		if layerType == selfAttentionLayer || crossAttentionStates != nil || cache.UnderlyingCache().(*kvcache.EncoderCache).EncoderCached() {
			var lastLayerOutputs ml.Tensor
			if i == len(d.Layers)-1 {
				lastLayerOutputs = outputs
			}

			hiddenState = layer.Forward(ctx, hiddenState, positionIDs, lastLayerOutputs, mask, crossAttentionStates, crossAttentionMask, cache, opts)
		}
	}

	return hiddenState
}

type TextModelOptions struct {
	RopeFactors ml.Tensor `gguf:"rope_freqs.weight"`

	hiddenSize, numHeads, numKVHeads int
	eps, ropeBase, ropeScale         float32
	ropeDim                          uint32

	crossAttentionLayers []uint32
}

type TextModel struct {
	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Transformer    *TextDecoder  `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output"`

	*TextModelOptions
}

func (m *TextModel) Forward(ctx ml.Context, inputIDs, positionIDs, outputs, mask, crossAttentionStates, crossAttentionMask ml.Tensor, cache *kvcache.WrapperCache) ml.Tensor {
	hiddenState := m.TokenEmbedding.Forward(ctx, inputIDs)
	hiddenState = m.Transformer.Forward(ctx, hiddenState, positionIDs, outputs, mask, crossAttentionStates, crossAttentionMask, cache, m.TextModelOptions)
	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	return m.Output.Forward(ctx, hiddenState)
}

func newTextModel(c ml.Config) *TextModel {
	var decoderLayers []TextDecoderLayer
	for i := range c.Uint("block_count") {
		var textDecoderLayer TextDecoderLayer
		if slices.Contains(c.Uints("attention.cross_attention_layers"), i) {
			textDecoderLayer = &TextCrossAttentionDecoderLayer{}
		} else {
			textDecoderLayer = &TextSelfAttentionDecoderLayer{}
		}

		decoderLayers = append(decoderLayers, textDecoderLayer)
	}

	return &TextModel{
		Transformer: &TextDecoder{Layers: decoderLayers},
		TextModelOptions: &TextModelOptions{
			hiddenSize:           int(c.Uint("embedding_length")),
			numHeads:             int(c.Uint("attention.head_count")),
			numKVHeads:           int(c.Uint("attention.head_count_kv")),
			eps:                  c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:             c.Float("rope.freq_base"),
			ropeScale:            c.Float("rope.freq_scale", 1),
			ropeDim:              c.Uint("rope.dimension_count"),
			crossAttentionLayers: c.Uints("attention.cross_attention_layers"),
		},
	}
}
