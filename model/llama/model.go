package llama

import (
	"math"

	"github.com/ollama/ollama/cache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
)

type Options struct {
	RopeFactors                      ml.Tensor `gguf:"rope_freqs.weight"`
	hiddenSize, numHeads, numKVHeads int64
	eps, ropeBase, ropeScale         float32
	ropeDim                          uint32
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
	return &Model{
		BytePairEncoding: model.BytePairEncoding{
			Pretokenizer: c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			Vocabulary: &model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Uints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				BOS:    c.Uint("tokenizer.ggml.bos_token_id"),
				EOS:    c.Uint("tokenizer.ggml.eos_token_id"),
			},
		},
		Layers: make([]Layer, c.Uint("block_count")),
		Options: &Options{
			hiddenSize: int64(c.Uint("embedding_length")),
			numHeads:   int64(c.Uint("attention.head_count")),
			numKVHeads: int64(c.Uint("attention.head_count_kv")),
			eps:        c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:   c.Float("rope.freq_base"),
			ropeScale:  c.Float("rope.freq_scale", 1),
			ropeDim:    c.Uint("rope.dimension_count"),
		},
	}, nil
}

type SelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache cache.Cache, opts *Options) ml.Tensor {
	// Ref: https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/llama.py
	// Batch size dimension(0) removed
	shape := hiddenState.Shape()
	L := shape[0]
	n_heads := opts.numHeads
	n_kv_heads := opts.numKVHeads
	head_dim := opts.hiddenSize / opts.numHeads

	queries := sa.Query.Forward(ctx, hiddenState)
	keys := sa.Key.Forward(ctx, hiddenState)
	values := sa.Value.Forward(ctx, hiddenState)

	queries = queries.Reshape(ctx, L, n_heads, -1).Permute(ctx, 0, 2, 1, 3)
	keys = keys.Reshape(ctx, L, n_kv_heads, -1).Permute(ctx, 0, 2, 1, 3)
	values = values.Reshape(ctx, L, n_kv_heads, -1).Permute(ctx, 0, 2, 1, 3)

	queries = LlamaRoPE(ctx, queries, positionIDs, opts)
	keys = LlamaRoPE(ctx, keys, positionIDs, opts)

	cache.Put(ctx, keys, values, 1)
	keys, values, mask := cache.Get(ctx)

	output := ScaledDotProductAttention(ctx, queries, keys, values, mask, float32(math.Pow(float64(head_dim), -0.5)))
	output = output.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	output = output.Reshape(ctx, L, -1)
	output = sa.Output.Forward(ctx, output)
	return output
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	g := mlp.Gate.Forward(ctx, hiddenState)
	x := mlp.Up.Forward(ctx, hiddenState)
	hiddenState = g.SILU(ctx).Mul(ctx, x)
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache cache.Cache, opts *Options) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, positionIDs, cache, opts)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	return hiddenState.Add(ctx, residual)
}

func (m *Model) Forward(ctx ml.Context, opts model.Options) (ml.Tensor, error) {
	inputs, err := ctx.FromIntSlice(opts.Inputs(), len(opts.Inputs()))
	if err != nil {
		return nil, err
	}
	positions, err := ctx.FromIntSlice(opts.Positions(), len(opts.Positions()))
	if err != nil {
		return nil, err
	}

	hiddenState := m.TokenEmbedding.Forward(ctx, inputs)
	// hiddenState = hiddenState.Reshape(ctx, 1, -1, 4096)

	for i, layer := range m.Layers {
		opts.Cache.SetLayer(i)
		hiddenState = layer.Forward(ctx, hiddenState, positions, opts.Cache, m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	hiddenState = m.Output.Forward(ctx, hiddenState)

	outputs, err := ctx.FromIntSlice(opts.Outputs(), len(opts.Outputs()))
	// outputs, err := ctx.FromIntSlice([]int32{int32(hiddenState.Dim(1)) - 1}, 1, 1)
	if err != nil {
		return nil, err
	}

	return hiddenState.Rows(ctx, outputs), nil
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return LlamaRoPE(ctx, key, shift, m.Options), nil
}

func init() {
	model.Register("llama", New)
}
