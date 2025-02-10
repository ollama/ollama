package gemma2

import (
	"math"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
)

type Options struct {
	RopeFactors                      ml.Tensor `gguf:"rope_freqs.weight"`
	hiddenSize, numHeads, numKVHeads int
	attnKeyLen, attnValLen           int
	eps, ropeBase, ropeScale         float32
	ropeDim                          uint32
}

type Model struct {
	model.Base
	model.SentencePieceModel

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`           // is this supposed to be root means square?
	Output         *nn.Linear    `gguf:"output,alt:token_embd"` // just set to token_embd?

	*Options
}

func New(c ml.Config) (model.Model, error) {
	m := Model{
		SentencePieceModel: model.NewSentencePieceModel(
			c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Scores: c.Floats("tokenizer.ggml.scores"),
				Types:  c.Uints("tokenizer.ggml.token_type"),
				BOS:    int32(c.Uint("tokenizer.ggml.bos_token_id")),
				EOS:    int32(c.Uint("tokenizer.ggml.eos_token_id")),
			},
		),
		Layers: make([]Layer, c.Uint("block_count")),
		Options: &Options{
			hiddenSize: int(c.Uint("embedding_length")),
			numHeads:   int(c.Uint("attention.head_count")),
			numKVHeads: int(c.Uint("attention.head_count_kv")),
			attnKeyLen: int(c.Uint("attention.key_length")),
			attnValLen: int(c.Uint("attention.value_length")),
			eps:        c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:   c.Float("rope.freq_base", 10000.0),
			ropeScale:  c.Float("rope.freq_scale", 1.0),
		},
	}

	slidingWindowLen := int32(c.Uint("attention.sliding_window"))
	m.Cache = kvcache.NewWrapperCache(kvcache.NewSWACache(slidingWindowLen, m.Shift), kvcache.NewCausalCache(m.Shift))

	return &m, nil
}

type SelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, opts.attnKeyLen, opts.numHeads, batchSize)
	q = q.RoPE(ctx, positionIDs, opts.RopeFactors, uint32(headDim), opts.ropeBase, opts.ropeScale)

	// todo: this should be 1.0/math.Sqrt(float64(headDim)) for 27B models
	//q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.attnKeyLen)))

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, opts.attnKeyLen, opts.numKVHeads, batchSize)
	k = k.RoPE(ctx, positionIDs, opts.RopeFactors, opts.ropeDim, opts.ropeBase, opts.ropeScale)

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, opts.attnValLen, opts.numKVHeads, batchSize)

	cache.Put(ctx, k, v)
	k, v, mask := cache.Get(ctx)

	q = q.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	k = k.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	v = v.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	kq := k.Mulmat(ctx, q)
	kq = kq.Scale(ctx, 1.0/math.Sqrt(float64(headDim)))
	kq = kq.Add(ctx, mask)
	kq = kq.Softmax(ctx)

	kqv := v.Mulmat(ctx, kq)
	kqv = kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	kqv = kqv.Reshape(ctx, opts.attnValLen*opts.numHeads, batchSize)

	return sa.Output.Forward(ctx, kqv)
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return key.RoPE(ctx, shift, m.Options.RopeFactors, m.Options.ropeDim, m.Options.ropeBase, m.Options.ropeScale), nil
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).Tanh(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
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
	inputs, err := ctx.FromIntSlice(opts.Inputs, len(opts.Inputs))
	if err != nil {
		return nil, err
	}
	inputs = inputs.Scale(ctx, math.Sqrt(float64(m.Options.hiddenSize)))

	positions, err := ctx.FromIntSlice(opts.Positions, len(opts.Positions))
	if err != nil {
		return nil, err
	}

	hiddenState := m.TokenEmbedding.Forward(ctx, inputs)
	ctx.Forward(hiddenState)

	for i, layer := range m.Layers {
		cacheType := i % 2
		m.Cache.SetLayer(i)
		wc := m.Cache.(*kvcache.WrapperCache)
		wc.SetLayerType(cacheType)
		hiddenState = layer.Forward(ctx, hiddenState, positions, m.Cache, m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	hiddenState = m.Output.Forward(ctx, hiddenState)

	outputs, err := ctx.FromIntSlice(opts.Outputs, len(opts.Outputs))
	if err != nil {
		return nil, err
	}

	return hiddenState.Rows(ctx, outputs), nil
}

func init() {
	model.Register("gemma2", New)
}
