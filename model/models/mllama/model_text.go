package mllama

import (
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

type TextSelfAttention struct {
	Query       *nn.Linear `gguf:"attn_q"`
	Key         *nn.Linear `gguf:"attn_k"`
	Value       *nn.Linear `gguf:"attn_v"`
	Output      *nn.Linear `gguf:"attn_output"`
	RopeFactors ml.Tensor  `gguf:"rope_freqs.weight"`
}

func (sa *TextSelfAttention) Forward(ctx ml.Context, hiddenState, positions ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads

	query := sa.Query.Forward(ctx, hiddenState)
	query = query.Reshape(ctx, headDim, opts.numHeads, batchSize)
	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions, sa.RopeFactors)

	key := sa.Key.Forward(ctx, hiddenState)
	key = key.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions, sa.RopeFactors)

	value := sa.Value.Forward(ctx, hiddenState)
	value = value.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	scaleFactor := 1.0 / math.Sqrt(float64(headDim))
	attention := nn.Attention(ctx, query, key, value, scaleFactor, cache)
	attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)

	return sa.Output.Forward(ctx, attention)
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	// This will only get called for layers in the cache, which are just the self attention layers
	if layer, ok := m.Transformer.Layers[layer].(*TextSelfAttentionDecoderLayer); ok {
		return m.applyRotaryPositionEmbeddings(ctx, key, shift, layer.SelfAttention.RopeFactors), nil
	}

	return key, nil
}

type TextMLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextModelOptions) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type TextSelfAttentionDecoderLayer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *TextSelfAttention

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP     *TextMLP
}

func (d *TextSelfAttentionDecoderLayer) Forward(ctx ml.Context, hiddenState, positions, outputs, _, _ ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	residual := hiddenState

	hiddenState = d.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = d.SelfAttention.Forward(ctx, hiddenState, positions, cache, opts)

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

	var key, value ml.Tensor
	if crossAttentionStates != nil {
		numVisionTokens, numTiles := crossAttentionStates.Dim(1), crossAttentionStates.Dim(2)

		key = ca.Key.Forward(ctx, crossAttentionStates)
		key = key.Reshape(ctx, headDim, opts.numKVHeads, numVisionTokens*numTiles)
		key = ca.KeyNorm.Forward(ctx, key, opts.eps)

		value = ca.Value.Forward(ctx, crossAttentionStates)
		value = value.Reshape(ctx, headDim, opts.numKVHeads, numVisionTokens*numTiles)

		cache.Put(ctx, key, value)
	}

	key, value, _ = cache.Get(ctx)

	scaleFactor := 1.0 / math.Sqrt(float64(headDim))

	query = query.Permute(ctx, 0, 2, 1, 3)
	key = key.Permute(ctx, 0, 2, 1, 3)
	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	kq := key.MulmatFullPrec(ctx, query)

	kq = kq.Scale(ctx, scaleFactor)
	kq = kq.Softmax(ctx)

	kqv := value.Mulmat(ctx, kq)
	attention := kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
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

func (d *TextCrossAttentionDecoderLayer) Forward(ctx ml.Context, hiddenState, _, _, crossAttentionStates, crossAttentionMask ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
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
	Forward(ctx ml.Context, hiddenState, positionIDs, outputs, crossAttentionStates, crossAttentionMask ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor
}

type TextDecoder struct {
	Layers []TextDecoderLayer
}

func (d *TextDecoder) Forward(ctx ml.Context, hiddenState, positionIDs, outputs, crossAttentionStates, crossAttentionMask ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	for i, layer := range d.Layers {
		layerType := selfAttentionLayer
		if slices.Contains(opts.crossAttentionLayers, int32(i)) {
			layerType = crossAttentionLayer
		}

		cache.SetLayer(i)
		cache.SetLayerType(layerType)

		if layerType == selfAttentionLayer || crossAttentionStates != nil || cache.UnderlyingCache().(*kvcache.EncoderCache).EncoderCached() {
			var lastLayerOutputs ml.Tensor
			if i == len(d.Layers)-1 {
				lastLayerOutputs = outputs
			}

			hiddenState = layer.Forward(ctx, hiddenState, positionIDs, lastLayerOutputs, crossAttentionStates, crossAttentionMask, cache, opts)
		}
	}

	return hiddenState
}

type TextModelOptions struct {
	hiddenSize, numHeads, numKVHeads int
	ropeDim                          int
	eps, ropeBase, ropeScale         float32

	crossAttentionLayers []int32
}

func (o TextModelOptions) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions, factors ml.Tensor) ml.Tensor {
	return nn.RoPE(ctx, states, positions, o.ropeDim, o.ropeBase, 1./o.ropeScale, rope.WithFactors(factors))
}

type TextModel struct {
	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Transformer    *TextDecoder  `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output"`

	*TextModelOptions
}

func (m *TextModel) Forward(ctx ml.Context, inputIDs, positionIDs, outputs, crossAttentionStates, crossAttentionMask ml.Tensor, cache *kvcache.WrapperCache) ml.Tensor {
	hiddenState := m.TokenEmbedding.Forward(ctx, inputIDs)
	hiddenState = m.Transformer.Forward(ctx, hiddenState, positionIDs, outputs, crossAttentionStates, crossAttentionMask, cache, m.TextModelOptions)
	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	return m.Output.Forward(ctx, hiddenState)
}

func newTextModel(c fs.Config) *TextModel {
	var decoderLayers []TextDecoderLayer
	for i := range c.Uint("block_count") {
		var textDecoderLayer TextDecoderLayer
		if slices.Contains(c.Ints("attention.cross_attention_layers"), int32(i)) {
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
			ropeDim:              int(c.Uint("rope.dimension_count")),
			eps:                  c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:             c.Float("rope.freq_base"),
			ropeScale:            c.Float("rope.scaling.factor", 1),
			crossAttentionLayers: c.Ints("attention.cross_attention_layers"),
		},
	}
}
