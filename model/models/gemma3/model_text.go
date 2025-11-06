package gemma3

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/ml/nn/rope"
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
			ropeScale:      1,
			// NOTE: the rope.scaling.factor is set incorrectly in the official QAT weights
			//       (8 instead of 1)
			// ropeScale:      c.Float("rope.scaling.factor", 1.0),
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

	ropeBase := opts.ropeLocalBase
	if (layer+1)%gemmaGlobalCacheCount == 0 {
		ropeBase = opts.ropeGlobalBase
	}

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, opts.attnKeyLen, opts.numHeads, batchSize)
	q = sa.QueryNorm.Forward(ctx, q, opts.eps)
	q = fast.RoPE(ctx, q, positionIDs, opts.attnKeyLen, ropeBase, 1./opts.ropeScale, rope.WithTypeNeoX())

	if opts.largeModelScaling {
		q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.hiddenSize/opts.numHeads)))
	} else {
		q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.attnKeyLen)))
	}

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, opts.attnKeyLen, opts.numKVHeads, batchSize)
	k = sa.KeyNorm.Forward(ctx, k, opts.eps)
	k = fast.RoPE(ctx, k, positionIDs, opts.attnKeyLen, ropeBase, 1./opts.ropeScale, rope.WithTypeNeoX())

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

	return fast.RoPE(ctx, key, shift, m.TextConfig.attnKeyLen, ropeBase, 1/m.TextConfig.ropeScale, rope.WithTypeNeoX()), nil
}

type TextMLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextConfig) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).GELU(ctx, mlp.Up.Forward(ctx, hiddenState))
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

func (m *TextModel) Forward(ctx ml.Context, batch input.Batch, cache kvcache.Cache) ml.Tensor {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	hiddenState = hiddenState.Scale(ctx, math.Sqrt(float64(m.TextConfig.hiddenSize)))

	// set image embeddings
	var except []int
	for _, image := range batch.Multimodal {
		visionOutputs := image.Multimodal[0].Tensor
		ctx.Forward(visionOutputs.Copy(ctx, hiddenState.View(ctx, image.Index*hiddenState.Stride(1), visionOutputs.Dim(0)*visionOutputs.Dim(1))))

		for i := range visionOutputs.Dim(1) {
			except = append(except, image.Index+i)
		}
	}

	for i, layer := range m.Layers {
		// gemma alternates between the sliding window (local) and causal (global)
		// kv cache every 6 layers
		if cache != nil {
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
		}

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = batch.Outputs
		}

		hiddenState = layer.Forward(ctx, i, hiddenState, positions, lastLayerOutputs, cache, m.TextConfig)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	return hiddenState
}
