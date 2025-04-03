package gemma2

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Options struct {
	hiddenSize, numHeads, numKVHeads int
	attnKeyLen, attnValLen           int
	eps, ropeBase, ropeScale         float32
	attnLogitSoftcap                 float32
	finalLogitSoftcap                float32
	largeModelScaling                bool
}

type Model struct {
	model.Base
	model.SentencePieceModel

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"` // just set to token_embd?

	*Options
}

const (
	gemma27BLayerCount = 46
)

func New(c fs.Config) (model.Model, error) {
	m := Model{
		SentencePieceModel: model.NewSentencePieceModel(
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
			hiddenSize:        int(c.Uint("embedding_length")),
			numHeads:          int(c.Uint("attention.head_count")),
			numKVHeads:        int(c.Uint("attention.head_count_kv")),
			attnKeyLen:        int(c.Uint("attention.key_length")),
			attnValLen:        int(c.Uint("attention.value_length")),
			eps:               c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:          c.Float("rope.freq_base", 10000.0),
			ropeScale:         c.Float("rope.freq_scale", 1.0),
			attnLogitSoftcap:  c.Float("attn_logit_softcapping"),
			finalLogitSoftcap: c.Float("final_logit_softcapping"),
		},
	}

	slidingWindowLen := int32(c.Uint("attention.sliding_window"))
	m.Cache = kvcache.NewWrapperCache(kvcache.NewSWACache(slidingWindowLen, m.Shift), kvcache.NewCausalCache(m.Shift))
	m.Cache.SetConfig(ml.CacheConfig{})

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
	ropeType := uint32(2)

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, opts.attnKeyLen, opts.numHeads, batchSize)
	q = q.RoPE(ctx, positionIDs, nil, uint32(opts.attnKeyLen), ropeType, opts.ropeBase, opts.ropeScale)

	if opts.largeModelScaling {
		q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.hiddenSize/opts.numHeads)))
	} else {
		q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.attnKeyLen)))
	}

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, opts.attnKeyLen, opts.numKVHeads, batchSize)
	k = k.RoPE(ctx, positionIDs, nil, uint32(opts.attnKeyLen), ropeType, opts.ropeBase, opts.ropeScale)

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, opts.attnValLen, opts.numKVHeads, batchSize)

	cache.Put(ctx, k, v)
	k, v, mask := cache.Get(ctx)

	q = q.Permute(ctx, 0, 2, 1, 3)
	k = k.Permute(ctx, 0, 2, 1, 3)
	v = v.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	kq := k.Mulmat(ctx, q)

	// logit softcap
	kq = kq.Scale(ctx, 1.0/float64(opts.attnLogitSoftcap))
	kq = kq.Tanh(ctx)
	kq = kq.Scale(ctx, float64(opts.attnLogitSoftcap))

	kq = kq.Add(ctx, mask)
	kq = kq.Softmax(ctx)

	kqv := v.Mulmat(ctx, kq)
	kqv = kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	kqv = kqv.Reshape(ctx, opts.attnValLen*opts.numHeads, batchSize)

	return sa.Output.Forward(ctx, kqv)
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return key.RoPE(ctx, shift, nil, uint32(m.Options.attnKeyLen), uint32(2), m.Options.ropeBase, m.Options.ropeScale), nil
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).GELU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenState))
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

func (l *Layer) Forward(ctx ml.Context, hiddenState, positionIDs, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, positionIDs, cache, opts)
	hiddenState = l.PostAttentionNorm.Forward(ctx, hiddenState, opts.eps)

	// In the final layer (outputs != nil), optimize by pruning to just the token positions
	// we need logits for.
	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	hiddenState = l.PostMLPNorm.Forward(ctx, hiddenState, opts.eps)
	return hiddenState.Add(ctx, residual)
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions, err := ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))
	if err != nil {
		return nil, err
	}

	outputs, err := ctx.Input().FromIntSlice(batch.Outputs, len(batch.Outputs))
	if err != nil {
		return nil, err
	}

	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	hiddenState = hiddenState.Scale(ctx, math.Sqrt(float64(m.Options.hiddenSize)))

	if len(m.Layers) == gemma27BLayerCount {
		m.Options.largeModelScaling = true
	}

	for i, layer := range m.Layers {
		cacheType := i % 2
		m.Cache.SetLayer(i)
		wc := m.Cache.(*kvcache.WrapperCache)
		wc.SetLayerType(cacheType)

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = outputs
		}

		hiddenState = layer.Forward(ctx, hiddenState, positions, lastLayerOutputs, m.Cache, m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	hiddenState = m.Output.Forward(ctx, hiddenState)

	// final logit softcap
	hiddenState = hiddenState.Scale(ctx, 1.0/float64(m.Options.finalLogitSoftcap))
	hiddenState = hiddenState.Tanh(ctx)
	return hiddenState.Scale(ctx, float64(m.Options.finalLogitSoftcap)), nil
}

func init() {
	model.Register("gemma2", New)
}
