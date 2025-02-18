package qwen2

import (
	"math"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
)

type Options struct {
	RopeFactors    ml.Tensor `gguf:"rope_freqs.weight"`
	contextLength  int
	hiddenSize     int
	numAttnHeads   int
	numKVHeads     int
	modelEpsilon   float32
	ropeBaseFreq   float32
	ropeFreqScale  float32
	ropeDimensions uint32
}

type Model struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*Options
}

func New(c ml.Config) (model.Model, error) {
	m := &Model{
		BytePairEncoding: model.NewBytePairEncoding(
			c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Uints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				BOS:    int32(c.Uint("tokenizer.ggml.bos_token_id")),
				EOS:    int32(c.Uint("tokenizer.ggml.eos_token_id")),
			},
		),
		Layers: make([]Layer, c.Uint("block_count")),
		Options: &Options{
			hiddenSize:     int(c.Uint("embedding_length")),
			numAttnHeads:   int(c.Uint("attention.head_count")),
			numKVHeads:     int(c.Uint("attention.head_count_kv")),
			modelEpsilon:   c.Float("attention.layer_norm_rms_epsilon"),
			contextLength:  int(c.Uint("context_length")),
			ropeBaseFreq:   c.Float("rope.freq_base"),
			ropeFreqScale:  c.Float("rope.freq_scale", 1),
			ropeDimensions: c.Uint("rope.dimension_count", 64),
		},
	}

	m.Cache = kvcache.NewCausalCache(m.Shift)

	return m, nil
}

// Shift applies rotary position embeddings to the key tensor for causal attention caching
func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return key.RoPE(
		ctx,
		ml.RopeConfig{
			PositionIDs: shift,
			RopeFactors: m.Options.RopeFactors,
			RopeDim:     m.Options.ropeDimensions,
			RopeType:    ml.RopeTypeNeoX,
			OrigCtxLen:  m.Options.contextLength,
			RopeBase:    m.Options.ropeBaseFreq,
			RopeScale:   m.Options.ropeFreqScale,
		},
	), nil
}

// SelfAttention implements the multi-head self-attention mechanism
// with separate projections for query, key, value and output transformations
type SelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, inputPositions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	// Initialize dimensions and configuration
	batchSize := hiddenState.Dim(1)
	headDimension := opts.hiddenSize / opts.numAttnHeads
	ropeConfig := ml.RopeConfig{
		PositionIDs: inputPositions,
		RopeFactors: nil,
		RopeDim:     opts.ropeDimensions,
		RopeType:    ml.RopeTypeNeoX,
		OrigCtxLen:  opts.contextLength,
		RopeBase:    opts.ropeBaseFreq,
		RopeScale:   opts.ropeFreqScale,
	}

	// Project and reshape query states with rotary embeddings
	queryStates := sa.Query.Forward(ctx, hiddenState)
	queryStates = queryStates.Reshape(ctx, headDimension, opts.numAttnHeads, batchSize)
	queryStates = queryStates.RoPE(ctx, ropeConfig)

	// Project and reshape key states with rotary embeddings
	keyStates := sa.Key.Forward(ctx, hiddenState)
	keyStates = keyStates.Reshape(ctx, headDimension, opts.numKVHeads, batchSize)
	keyStates = keyStates.RoPE(ctx, ropeConfig)

	// Project and reshape value states
	valueStates := sa.Value.Forward(ctx, hiddenState)
	valueStates = valueStates.Reshape(ctx, headDimension, opts.numKVHeads, batchSize)

	// Update and retrieve from KV cache
	cache.Put(ctx, keyStates, valueStates)
	keyStates, valueStates, attentionMask := cache.Get(ctx)

	// Prepare tensors for attention computation
	queryStates = queryStates.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	keyStates = keyStates.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	valueStates = valueStates.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	// Apply scaling and attention mask to scores
	attentionScores := keyStates.MulmatFullPrec(ctx, queryStates)
	attentionScores = attentionScores.Scale(ctx, 1.0/math.Sqrt(float64(headDimension)))
	attentionScores = attentionScores.Add(ctx, attentionMask)
	// Compute scaled dot-product attention
	attentionProbs := attentionScores.Softmax(ctx)

	// Apply attention weights and reshape
	weightedStates := valueStates.Mulmat(ctx, attentionProbs)
	weightedStates = weightedStates.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	weightedStates = weightedStates.Reshape(ctx, opts.hiddenSize, batchSize)

	// Project to output dimension
	return sa.Output.Forward(ctx, weightedStates)
}

// MLP implements the feed-forward network component with SwiGLU activation
type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	// Apply SwiGLU activation gating
	gateActivation := mlp.Gate.Forward(ctx, hiddenState).SILU(ctx)
	upProjection := mlp.Up.Forward(ctx, hiddenState)
	intermediateStates := gateActivation.Mul(ctx, upProjection)

	// Project back to hidden dimension
	return mlp.Down.Forward(ctx, intermediateStates)
}

// Layer represents a single transformer layer combining self-attention and feed-forward components
type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	// Self-attention branch with residual connection
	residual := hiddenState

	normalizedAttention := l.AttentionNorm.Forward(ctx, hiddenState, opts.modelEpsilon)
	attentionOutput := l.SelfAttention.Forward(ctx, normalizedAttention, positionIDs, cache, opts)
	hiddenState = attentionOutput.Add(ctx, residual)

	// Feed-forward branch with residual connection
	residual = hiddenState
	normalizedMLP := l.MLPNorm.Forward(ctx, hiddenState, opts.modelEpsilon)
	mlpOutput := l.MLP.Forward(ctx, normalizedMLP, opts)
	output := mlpOutput.Add(ctx, residual)

	return output
}

func (m *Model) Forward(ctx ml.Context, opts model.Options) (ml.Tensor, error) {
	// Convert input tokens and positions to tensors
	inputTensor, err := ctx.FromIntSlice(opts.Inputs, len(opts.Inputs))
	if err != nil {
		return nil, err
	}

	positionsTensor, err := ctx.FromIntSlice(opts.Positions, len(opts.Positions))
	if err != nil {
		return nil, err
	}

	// Initial token embedding
	hiddenStates := m.TokenEmbedding.Forward(ctx, inputTensor)

	// Process through transformer layers
	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)
		hiddenStates = layer.Forward(ctx, hiddenStates, positionsTensor, m.Cache, m.Options)
	}

	// Final layer normalization and output projection
	normalizedOutput := m.OutputNorm.Forward(ctx, hiddenStates, m.modelEpsilon)
	logits := m.Output.Forward(ctx, normalizedOutput)

	// Extract requested output token positions
	outputsTensor, err := ctx.FromIntSlice(opts.Outputs, len(opts.Outputs))
	if err != nil {
		return nil, err
	}

	return logits.Rows(ctx, outputsTensor), nil
}

func init() {
	model.Register("qwen2", New)
}
