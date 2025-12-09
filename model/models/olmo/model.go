package olmo

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

const (
	cacheTypeSWA = iota
	cacheTypeCausal
)

type Options struct {
	hiddenSize, numHeads, numKVHeads int
	// headDim, ropeDim                 int
	eps, ropeBase, ropeScale float32

	originalContextLength int
	attnFactor            float32

	ropeType          string
	ropeExtrapolation float32
	ropeBetaFast      float32
	ropeBetaSlow      float32

	slidingWindowPattern []bool
}

type Model struct {
	model.Base
	model.TextProcessor

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	Options
}

func New(c fs.Config) (model.Model, error) {
	vocabulary := model.Vocabulary{
		Values: c.Strings("tokenizer.ggml.tokens"),
		Scores: c.Floats("tokenizer.ggml.scores"),
		Types:  c.Ints("tokenizer.ggml.token_type"),
		Merges: c.Strings("tokenizer.ggml.merges"),
		AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
		BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
		AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
		EOS: append(
			[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
			c.Ints("tokenizer.ggml.eos_token_ids")...,
		),
	}

	if c.String("tokenizer.ggml.model") != "gpt2" {
		return nil, model.ErrUnsupportedTokenizer
	}

	var pretokenizers []string
	if c.String("tokenizer.ggml.pre") != "default" {
		pretokenizers = []string{
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`,
		}
	}
	processor := model.NewBytePairEncoding(&vocabulary, pretokenizers...)

	hiddenSize := int(c.Uint("embedding_length"))
	numHeads := int(c.Uint("attention.head_count"))
	numKVHeads := int(c.Uint("attention.head_count_kv"))
	// headDim := int(c.Uint("attention.head_count"))
	// ropeDim := int(c.Uint("rope.dimension_count"))
	eps := c.Float("attention.layer_norm_rms_epsilon")
	ropeBase := c.Float("rope.freq_base", 1e4)
	ropeScale := c.Float("rope.scaling.factor", 1)
	originalContextLength := int(c.Uint("rope.scaling.original_context_length"))
	attnFactor := c.Float("rope.scaling.attn_factor", 1)
	ropeType := c.String("rope.scaling.type")
	ropeExtrapolation := c.Float("rope.scaling.extrapolation_factor", 1.0)
	ropeBetaFast := c.Float("rope.scaling.beta_fast", 64.0)
	ropeBetaSlow := c.Float("rope.scaling.beta_slow", 1.0)

	fmt.Printf("hiddenSize: %d\n", hiddenSize)
	fmt.Printf("numHeads: %d\n", numHeads)
	fmt.Printf("numKVHeads: %d\n", numKVHeads)
	// fmt.Printf("headDim: %d\n", headDim)
	// fmt.Printf("ropeDim: %d\n", ropeDim)
	fmt.Printf("eps: %f\n", eps)
	fmt.Printf("ropeBase: %f\n", ropeBase)
	fmt.Printf("ropeScale: %f\n", ropeScale)
	fmt.Printf("originalContextLength: %d\n", originalContextLength)
	fmt.Printf("attnFactor: %f\n", attnFactor)
	fmt.Printf("ropeType: %s\n", ropeType)
	fmt.Printf("ropeExtrapolation: %f\n", ropeExtrapolation)
	fmt.Printf("ropeBetaFast: %f\n", ropeBetaFast)
	fmt.Printf("ropeBetaSlow: %f\n", ropeBetaSlow)
	fmt.Printf("sliding_window_pattern: %v\n", c.Bools("attention.sliding_window_pattern"))

	m := Model{
		TextProcessor: processor,
		Layers:        make([]Layer, c.Uint("block_count")),
		Options: Options{
			hiddenSize: hiddenSize,
			numHeads:   numHeads,
			numKVHeads: numKVHeads,
			// headDim:               headDim,
			// ropeDim:               ropeDim,
			eps:                   eps,
			ropeBase:              ropeBase,
			ropeScale:             ropeScale,
			originalContextLength: originalContextLength,
			attnFactor:            attnFactor,
			ropeType:              ropeType,
			ropeExtrapolation:     ropeExtrapolation,
			ropeBetaFast:          ropeBetaFast,
			ropeBetaSlow:          ropeBetaSlow,
			slidingWindowPattern:  c.Bools("attention.sliding_window_pattern"),
		},
	}

	m.Cache = kvcache.NewWrapperCache(kvcache.NewSWACache(int32(c.Uint("attention.sliding_window")), m.Shift), kvcache.NewCausalCache(m.Shift))
	// m.Cache = kvcache.NewCausalCache(m.Shift)

	return &m, nil
}

type SelfAttention struct {
	Query       *nn.Linear  `gguf:"attn_q"`
	Key         *nn.Linear  `gguf:"attn_k"`
	Value       *nn.Linear  `gguf:"attn_v"`
	Output      *nn.Linear  `gguf:"attn_output"`
	QNorm       *nn.RMSNorm `gguf:"attn_q_norm"`
	KNorm       *nn.RMSNorm `gguf:"attn_k_norm"`
	RopeFactors ml.Tensor   `gguf:"rope_freqs.weight"`
}

func (o *Options) ropeOptions(factors ml.Tensor, isSWA bool) []func(*rope.Options) {
	opts := []func(*rope.Options){
		rope.WithFactors(factors),
	}

	if !isSWA && o.originalContextLength > 0 {
		// opts = append(opts,
		// 	rope.WithOriginalContextLength(o.originalContextLength),
		// 	rope.WithAttentionFactor(o.attnFactor),
		// )
		opts = append(opts,
			rope.WithOriginalContextLength(o.originalContextLength),
			rope.WithExtrapolationFactor(o.ropeExtrapolation),
			rope.WithAttentionFactor(o.attnFactor),
			rope.WithBetaFast(o.ropeBetaFast),
			rope.WithBetaSlow(o.ropeBetaSlow),
		)
	} else if isSWA && o.originalContextLength > 0 {
		opts = append(opts,
			rope.WithOriginalContextLength(o.originalContextLength),
			rope.WithExtrapolationFactor(0.),
			rope.WithAttentionFactor(1.),
		)
	}

	return opts
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positions ml.Tensor, cache kvcache.Cache, opts *Options, isSWA bool) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads
	ropeDim := headDim

	query := sa.Query.Forward(ctx, hiddenState)
	if sa.QNorm != nil {
		query = sa.QNorm.Forward(ctx, query, opts.eps)
	}
	query = query.Reshape(ctx, headDim, opts.numHeads, batchSize)

	key := sa.Key.Forward(ctx, hiddenState)
	if sa.KNorm != nil {
		key = sa.KNorm.Forward(ctx, key, opts.eps)
	}
	key = key.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	value := sa.Value.Forward(ctx, hiddenState)
	value = value.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	freqScale := float32(1.0)
	if !isSWA {
		freqScale = 1. / opts.ropeScale
	}

	ropeOpts := opts.ropeOptions(sa.RopeFactors, isSWA)
	query = nn.RoPE(ctx, query, positions, ropeDim, opts.ropeBase, freqScale, ropeOpts...)
	key = nn.RoPE(ctx, key, positions, ropeDim, opts.ropeBase, freqScale, ropeOpts...)

	attention := nn.Attention(ctx, query, key, value, 1.0/math.Sqrt(float64(headDim)), cache)
	attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)

	return sa.Output.Forward(ctx, attention)
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeDim := m.hiddenSize / m.numHeads
	isSWA := m.isSWALayer(layer)

	freqScale := float32(1.0)
	if !isSWA {
		freqScale = 1. / m.ropeScale
	}

	ropeOpts := m.Options.ropeOptions(m.Layers[layer].SelfAttention.RopeFactors, isSWA)
	return nn.RoPE(ctx, key, shift, ropeDim, m.ropeBase, freqScale, ropeOpts...), nil
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	SelfAttention     *SelfAttention
	PostAttentionNorm *nn.RMSNorm `gguf:"post_attention_norm"`
	MLP               *MLP
	PostFFWNorm       *nn.RMSNorm `gguf:"post_ffw_norm"`
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options, isSWA bool) ml.Tensor {
	residual := hiddenState

	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, positions, cache, opts, isSWA)
	if l.PostAttentionNorm != nil {
		hiddenState = l.PostAttentionNorm.Forward(ctx, hiddenState, opts.eps)
	}

	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	hiddenState = l.PostFFWNorm.Forward(ctx, hiddenState, opts.eps)

	return hiddenState.Add(ctx, residual)
}

// Olmo3 has Sliding Window Attention (SWA) 3 out of 4 layers.
func (m *Model) isSWALayer(layerIdx int) bool {
	return m.Options.slidingWindowPattern[layerIdx]
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	hiddenState = hiddenState.Scale(ctx, math.Sqrt(float64(m.hiddenSize)))

	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)
		cacheType := cacheTypeSWA

		isSWA := m.isSWALayer(i)
		if !isSWA {
			cacheType = cacheTypeCausal
		}

		if wc, ok := m.Cache.(*kvcache.WrapperCache); ok {
			wc.SetLayerType(cacheType)
		}
		if causal, ok := m.Cache.(*kvcache.Causal); ok {
			causal.SetCausal(ctx, kvcache.CausalOptions{Except: []int{i}})
		}

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenState = layer.Forward(ctx, hiddenState, positions, outputs, m.Cache, &m.Options, isSWA)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	return m.Output.Forward(ctx, hiddenState), nil
}

func init() {
	model.Register("olmo2", New)
}
