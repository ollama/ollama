package rnj1

import (
	"cmp"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Options struct {
	hiddenSize                 int
	numHeads, numKVHeads       int
	headDim                    int
	eps                        float32
	ropeBase, ropeScale        float32
	ropeType                   string
	ropeOriginalContextLength  int
	ropeExtrapolationFactor    float32
	ropeBetaFast, ropeBetaSlow float32
	attentionScale             float32
}

func (o *Options) headDimSize() int {
	return cmp.Or(o.headDim, o.hiddenSize/o.numHeads)
}

func (o *Options) ropeOptions() []func(*rope.Options) {
	opts := []func(*rope.Options){rope.WithTypeNeoX()}
	if o.ropeType == "yarn" {
		attnFactor := float32(1.0 / (1.0 + 0.1*math.Log(float64(o.ropeScale))))
		opts = append(opts,
			rope.WithOriginalContextLength(o.ropeOriginalContextLength),
			rope.WithExtrapolationFactor(o.ropeExtrapolationFactor),
			rope.WithAttentionFactor(attnFactor),
			rope.WithBetaFast(o.ropeBetaFast),
			rope.WithBetaSlow(o.ropeBetaSlow),
		)
	}
	return opts
}

func (o *Options) applyRoPE(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	return fast.RoPE(ctx, states, positions, o.headDimSize(), o.ropeBase, 1./o.ropeScale, o.ropeOptions()...)
}

type SelfAttention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.headDimSize()

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, headDim, opts.numHeads, batchSize)
	q = sa.QueryNorm.Forward(ctx, q, opts.eps)
	q = opts.applyRoPE(ctx, q, positions)

	// Apply attention scaling (from hparams.f_attention_scale)
	q = q.Scale(ctx, float64(opts.attentionScale))

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	k = sa.KeyNorm.Forward(ctx, k, opts.eps)
	k = opts.applyRoPE(ctx, k, positions)

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	// Use scale factor of 1.0 since we already applied attention scaling to Q
	attention := nn.Attention(ctx, q, k, v, 1.0, cache)
	attention = attention.Reshape(ctx, headDim*opts.numHeads, batchSize)

	return sa.Output.Forward(ctx, attention)
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	// GELU activation (gate * gelu(up))
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).GELU(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm     *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention     *SelfAttention
	PostAttentionNorm *nn.RMSNorm `gguf:"post_attention_norm"`
	MLPNorm           *nn.RMSNorm `gguf:"ffn_norm"`
	MLP               *MLP
	PostMLPNorm       *nn.RMSNorm `gguf:"post_ffw_norm"`
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenState

	// Pre-attention norm
	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)

	// Self-attention
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, positions, cache, opts)

	// Post-attention norm
	hiddenState = l.PostAttentionNorm.Forward(ctx, hiddenState, opts.eps)

	// Optimize on final layer: prune to just the token positions we need logits for
	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	// First residual connection (after attention)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	// Pre-FFN norm
	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)

	// Feed-forward network
	hiddenState = l.MLP.Forward(ctx, hiddenState)

	// Post-FFN norm
	hiddenState = l.PostMLPNorm.Forward(ctx, hiddenState, opts.eps)

	// Second residual connection (after FFN)
	return hiddenState.Add(ctx, residual)
}

type Model struct {
	model.Base
	model.TextProcessor

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*Options
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	hiddenState = hiddenState.Scale(ctx, math.Sqrt(float64(m.hiddenSize)))

	for i, layer := range m.Layers {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
		}

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = batch.Outputs
		}

		hiddenState = layer.Forward(ctx, hiddenState, positions, lastLayerOutputs, m.Cache, m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)

	return m.Output.Forward(ctx, hiddenState), nil
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRoPE(ctx, key, shift), nil
}

func New(c fs.Config) (model.Model, error) {
	var processor model.TextProcessor
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
	switch c.String("tokenizer.ggml.model") {
	case "gpt2":
		// Use default BPE pretokenizer
		processor = model.NewBytePairEncoding(&vocabulary)
	case "llama":
		processor = model.NewSentencePiece(&vocabulary)
	default:
		return nil, model.ErrUnsupportedTokenizer
	}

	m := Model{
		TextProcessor: processor,
		Layers:        make([]Layer, c.Uint("block_count")),
		Options: &Options{
			hiddenSize:                int(c.Uint("embedding_length")),
			numHeads:                  int(c.Uint("attention.head_count", 8)),
			numKVHeads:                int(c.Uint("attention.head_count_kv", 4)),
			headDim:                   int(c.Uint("attention.key_length", 128)),
			eps:                       c.Float("attention.layer_norm_rms_epsilon", 1e-06),
			ropeBase:                  c.Float("rope.freq_base", 10000.0),
			ropeScale:                 c.Float("rope.scaling.factor", 1.0),
			ropeType:                  c.String("rope.scaling.type"),
			ropeOriginalContextLength: int(c.Uint("rope.scaling.original_context_length")),
			ropeExtrapolationFactor:   c.Float("rope.scaling.extrapolation_factor", 1.0),
			ropeBetaFast:              c.Float("rope.scaling.beta_fast", 64.0),
			ropeBetaSlow:              c.Float("rope.scaling.beta_slow", 1.0),
			attentionScale:            c.Float("attention.scale", 1.0/float32(math.Sqrt(float64(c.Uint("attention.key_length", 128))))),
		},
	}

	// Use causal cache (all layers are global, no sliding window)
	m.Cache = kvcache.NewCausalCache(m.Shift)

	return &m, nil
}

func init() {
	model.Register("rnj1", New)
}
