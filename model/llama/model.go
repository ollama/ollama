package llama

import (
	"math"

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

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache model.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, headDim, opts.numHeads, batchSize)
	q = q.RoPE(ctx, positionIDs, opts.RopeFactors, opts.ropeDim, opts.ropeBase, opts.ropeScale)

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	k = k.RoPE(ctx, positionIDs, opts.RopeFactors, opts.ropeDim, opts.ropeBase, opts.ropeScale)

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	k, v = cache.Put(ctx, k, v, cache.Options)

	q = q.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	k = k.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	v = v.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	kq := k.Mulmat(ctx, q)
	kq = kq.Scale(ctx, 1.0/math.Sqrt(float64(headDim)))
	kq = kq.Softmax(ctx)

	kqv := v.Mulmat(ctx, kq)
	kqv = kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	kqv = kqv.Reshape(ctx, opts.hiddenSize, batchSize)

	return sa.Output.Forward(ctx, kqv)
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache model.Cache, opts *Options) ml.Tensor {
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

	for i, layer := range m.Layers {
		hiddenState = layer.Forward(ctx, hiddenState, positions, opts.Cache.Sub(i), m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	hiddenState = m.Output.Forward(ctx, hiddenState)

	outputs, err := ctx.FromIntSlice([]int32{int32(len(opts.Positions())) - 1}, 1)
	if err != nil {
		return nil, err
	}

	return hiddenState.Rows(ctx, outputs), nil
}

func init() {
	model.Register("llama", New)
}
