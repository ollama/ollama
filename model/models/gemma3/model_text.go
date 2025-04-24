package gemma3

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type TextConfig struct {
	hiddenSize, numHeads, numKVHeads int
	attnKeyLen, attnValLen           int
	eps, ropeScale                   float32
	ropeLocalBase, ropeGlobalBase    float32
	largeModelScaling                bool
}

type TextModel struct {
	model.Base
	model.SentencePieceModel

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []TextLayer   `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*TextConfig
}

const (
	gemmaGlobalCacheCount = 6
	gemma27BLayerCount    = 62
)

const (
	cacheTypeSWA = iota
	cacheTypeCausal
)

func newTextModel(c fs.Config) *TextModel {
	numBlocks := int(c.Uint("block_count"))

	m := TextModel{
		SentencePieceModel: model.NewSentencePieceModel(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Scores: c.Floats("tokenizer.ggml.scores"),
				Types:  c.Uints("tokenizer.ggml.token_type"),
				BOS:    int32(c.Uint("tokenizer.ggml.bos_token_id")),
				EOS:    int32(c.Uint("tokenizer.ggml.eos_token_id")),
			},
		),
		Layers: make([]TextLayer, numBlocks),
		TextConfig: &TextConfig{
			hiddenSize:     int(c.Uint("embedding_length")),
			numHeads:       int(c.Uint("attention.head_count")),
			numKVHeads:     int(c.Uint("attention.head_count_kv")),
			attnKeyLen:     int(c.Uint("attention.key_length", 256)),
			attnValLen:     int(c.Uint("attention.value_length", 256)),
			eps:            c.Float("attention.layer_norm_rms_epsilon", 1e-06),
			ropeLocalBase:  c.Float("rope.local.freq_base", 10000.0),
			ropeGlobalBase: c.Float("rope.global.freq_base", 1000000.0),
			ropeScale:      c.Float("rope.freq_scale", 1.0),
		},
	}

	if numBlocks == gemma27BLayerCount {
		m.largeModelScaling = true
	}

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

func (sa *TextSelfAttention) Forward(ctx ml.Context, layer int, hiddenState, positionIDs ml.Tensor, cache kvcache.Cache, opts *TextConfig) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	ropeType := uint32(2)

	ropeBase := opts.ropeLocalBase
	if (layer+1)%gemmaGlobalCacheCount == 0 {
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
	ropeBase := m.TextConfig.ropeLocalBase
	if (layer+1)%gemmaGlobalCacheCount == 0 {
		ropeBase = m.TextConfig.ropeGlobalBase
	}

	return key.RoPE(ctx, shift, nil, uint32(m.TextConfig.attnKeyLen), uint32(2), ropeBase, m.TextConfig.ropeScale), nil
}

type TextMLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextConfig) ml.Tensor {
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

func (l *TextLayer) Forward(ctx ml.Context, layer int, hiddenState, positionIDs, outputs ml.Tensor, cache kvcache.Cache, opts *TextConfig) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, layer, hiddenState, positionIDs, cache, opts)
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

func (m *TextModel) Forward(ctx ml.Context, inputs, positions, outputs ml.Tensor, batch input.Batch, cache kvcache.Cache) ml.Tensor {
	hiddenState := m.TokenEmbedding.Forward(ctx, inputs)
	hiddenState = hiddenState.Scale(ctx, math.Sqrt(float64(m.TextConfig.hiddenSize)))

	// set image embeddings
	var except []int
	for _, image := range batch.Multimodal {
		visionOutputs := image.Multimodal.(ml.Tensor)
		ctx.Forward(visionOutputs.Copy(ctx, hiddenState.View(ctx, image.Index*hiddenState.Stride(1), visionOutputs.Dim(0)*visionOutputs.Dim(1))))

		for i := range visionOutputs.Dim(1) {
			except = append(except, image.Index+i)
		}
	}

	for i, layer := range m.Layers {
		// gemma alternates between the sliding window (local) and causal (global)
		// kv cache every 6 layers
		cacheType := cacheTypeSWA
		if (i+1)%gemmaGlobalCacheCount == 0 {
			cacheType = cacheTypeCausal
		}
		cache.SetLayer(i)
		wc := cache.(*kvcache.WrapperCache)
		wc.SetLayerType(cacheType)

		if causal, ok := wc.UnderlyingCache().(*kvcache.Causal); ok {
			causal.SetCausal(ctx, kvcache.CausalOptions{Except: except})
		}

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = outputs
		}

		hiddenState = layer.Forward(ctx, i, hiddenState, positions, lastLayerOutputs, cache, m.TextConfig)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	return m.Output.Forward(ctx, hiddenState)
}
