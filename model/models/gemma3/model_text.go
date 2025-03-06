package gemma3

import (
	"math"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
)

type TextOptions struct {
	hiddenSize, numHeads, numKVHeads int
	attnKeyLen, attnValLen           int
	eps, ropeScale                   float32
	ropeLocalBase, ropeGlobalBase    float32
	finalLogitSoftcap                float32
	largeModelScaling                bool
}

type TextModel struct {
	model.Base
	model.SentencePieceModel

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []TextLayer   `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*TextOptions
}

const (
	gemma27BLayerCount = 46
)

const (
	cacheTypeSWA = iota
	cacheTypeCausal
)

func newTextModel(c ml.Config) *TextModel {
	m := TextModel{
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
		Layers: make([]TextLayer, c.Uint("block_count")),
		TextOptions: &TextOptions{
			hiddenSize:        int(c.Uint("embedding_length")),
			numHeads:          int(c.Uint("attention.head_count")),
			numKVHeads:        int(c.Uint("attention.head_count_kv")),
			attnKeyLen:        int(c.Uint("attention.key_length")),
			attnValLen:        int(c.Uint("attention.value_length")),
			eps:               c.Float("text.attention.layer_norm_rms_epsilon"),
			ropeLocalBase:     c.Float("text.rope.local.freq_base", 10000.0),
			ropeGlobalBase:    c.Float("text.rope.global.freq_base", 1000000.0),
			ropeScale:         c.Float("text.rope.freq_scale", 1.0),
			finalLogitSoftcap: c.Float("text.final_logit_softcapping"),
		},
	}

	slidingWindowLen := int32(c.Uint("text.attention.sliding_window"))
	m.Cache = kvcache.NewWrapperCache(kvcache.NewSWACache(slidingWindowLen, m.Shift), kvcache.NewCausalCache(m.Shift))

	return &m
}

type TextSelfAttention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *TextSelfAttention) Forward(ctx ml.Context, layer int, hiddenState, positionIDs ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	ropeType := uint32(2)

	ropeBase := opts.ropeLocalBase
	if (layer+1)%6 == 0 {
		ropeBase = opts.ropeGlobalBase
	}

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, opts.attnKeyLen, opts.numHeads, batchSize)
	q = sa.QueryNorm.Forward(ctx, q, opts.eps)
	q = q.RoPE(ctx, positionIDs, nil, uint32(opts.attnKeyLen), ropeType, ropeBase, opts.ropeScale)

	if opts.largeModelScaling {
		q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.hiddenSize/opts.numHeads)))
	} else {
		q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.attnKeyLen)))
	}

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, opts.attnKeyLen, opts.numKVHeads, batchSize)
	k = sa.KeyNorm.Forward(ctx, k, opts.eps)
	k = k.RoPE(ctx, positionIDs, nil, uint32(opts.attnKeyLen), ropeType, ropeBase, opts.ropeScale)

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, opts.attnValLen, opts.numKVHeads, batchSize)

	scaleFactor := 1.0
	kqv := nn.Attention(ctx, q, k, v, scaleFactor, cache)
	kqv = kqv.Reshape(ctx, opts.attnValLen*opts.numHeads, batchSize)

	return sa.Output.Forward(ctx, kqv)
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeBase := m.TextOptions.ropeLocalBase
	if (layer+1)%6 == 0 {
		ropeBase = m.TextOptions.ropeGlobalBase
	}

	return key.RoPE(ctx, shift, nil, uint32(m.TextOptions.attnKeyLen), uint32(2), ropeBase, m.TextOptions.ropeScale), nil
}

type TextMLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextOptions) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).GELU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type TextLayer struct {
	AttentionNorm     *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention     *TextSelfAttention
	PostAttentionNorm *nn.RMSNorm `gguf:"post_attention_norm"`
	MLPNorm           *nn.RMSNorm `gguf:"ffn_norm"`
	MLP               *TextMLP
	PostMLPNorm       *nn.RMSNorm `gguf:"post_ffw_norm"`
}

func (l *TextLayer) Forward(ctx ml.Context, layer int, hiddenState, positionIDs ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, layer, hiddenState, positionIDs, cache, opts)
	hiddenState = l.PostAttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	hiddenState = l.PostMLPNorm.Forward(ctx, hiddenState, opts.eps)
	return hiddenState.Add(ctx, residual)
}

func (m *TextModel) Forward(ctx ml.Context, inputs, positions, embeddings, outputs ml.Tensor, cache kvcache.Cache) ml.Tensor {
	if embeddings == nil {
		embeddings = m.TokenEmbedding.Forward(ctx, inputs)
	}

	hiddenState := embeddings.Scale(ctx, math.Sqrt(float64(m.TextOptions.hiddenSize)))

	if len(m.Layers) == gemma27BLayerCount {
		m.TextOptions.largeModelScaling = true
	}

	for i, layer := range m.Layers {
		// gemma alternates between the sliding window (local) and causal (global)
		// kv cache every 6 layers
		cacheType := cacheTypeSWA
		if (i+1)%6 == 0 {
			cacheType = cacheTypeCausal
		}
		cache.SetLayer(i)
		wc := cache.(*kvcache.WrapperCache)
		wc.SetLayerType(cacheType)
		hiddenState = layer.Forward(ctx, i, hiddenState, positions, cache, m.TextOptions)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	hiddenState = m.Output.Forward(ctx, hiddenState)

	// final logit softcap
	hiddenState = hiddenState.Scale(ctx, 1.0/float64(m.TextOptions.finalLogitSoftcap))
	hiddenState = hiddenState.Tanh(ctx)
	hiddenState = hiddenState.Scale(ctx, float64(m.TextOptions.finalLogitSoftcap))

	return hiddenState.Rows(ctx, outputs)
}
