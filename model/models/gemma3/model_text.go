package gemma3

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model/input"
)

type TextConfig struct {
	hiddenSize, contextLength, numHeads, numKVHeads int
	attnKeyLen, attnValLen                          int
	eps, ropeScale                                  float32
	ropeLocalBase                                   float32
	largeModelScaling                               bool
	slidingWindow                                   uint32
	slidingWindowPattern                            []bool
	ropeBase                                        float32
	ropeType                                        string
	ropeOriginalContext                             int
	ropeExtrapolation                               float32
	ropeBetaFast                                    float32
	ropeBetaSlow                                    float32
	finalLogitSoftcap                               float32
}

func (o TextConfig) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor, base, scale float32) ml.Tensor {
	ropeOpts := []func(*rope.Options){rope.WithTypeNeoX()}
	if o.ropeType == "yarn" {
		attnFactor := float32(1.0 / (1.0 + 0.1*math.Log(float64(scale))))
		ropeOpts = append(ropeOpts,
			rope.WithOriginalContextLength(o.ropeOriginalContext),
			rope.WithExtrapolationFactor(o.ropeExtrapolation),
			rope.WithAttentionFactor(attnFactor),
			rope.WithBetaFast(o.ropeBetaFast),
			rope.WithBetaSlow(o.ropeBetaSlow),
		)
	}

	return nn.RoPE(ctx, states, positions, o.attnKeyLen, base, 1./scale, ropeOpts...)
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
	gemma1BLayerCount     = 26
	gemma4BLayerCount     = 34
	gemma12BLayerCount    = 48
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
			hiddenSize:           int(c.Uint("embedding_length")),
			contextLength:        int(c.Uint("context_length")),
			numHeads:             int(c.Uint("attention.head_count")),
			numKVHeads:           int(c.Uint("attention.head_count_kv")),
			attnKeyLen:           int(c.Uint("attention.key_length", 256)),
			attnValLen:           int(c.Uint("attention.value_length", 256)),
			eps:                  c.Float("attention.layer_norm_rms_epsilon", 1e-06),
			ropeLocalBase:        c.Float("rope.local.freq_base", 10000.0),
			ropeBase:             c.Float("rope.freq_base", 1000000.0),
			slidingWindow:        c.Uint("attention.sliding_window"),
			slidingWindowPattern: c.Bools("attention.sliding_window_pattern"),
			ropeType:             c.String("rope.scaling.type"),
			ropeOriginalContext:  int(c.Uint("rope.scaling.original_context_length")),
			ropeExtrapolation:    c.Float("rope.scaling.extrapolation_factor", 1.0),
			ropeBetaFast:         c.Float("rope.scaling.beta_fast", 64.0),
			ropeBetaSlow:         c.Float("rope.scaling.beta_slow", 1.0),
			ropeScale:            c.Float("rope.scaling.factor", 1.0),
			finalLogitSoftcap:    c.Float("final_logit_softcapping", 0.0),
		},
	}

	// Apply corrections for older versions of the Gemma 3 models
	// by looking at whether they use sliding window attention and
	// based on their layer counts.
	if m.TextConfig.slidingWindow < uint32(m.TextConfig.contextLength) {
		switch numBlocks {
		case gemma1BLayerCount:
			// The 1B model has final logit softcapping set to 30.0
			// but it should be 0.0
			m.TextConfig.finalLogitSoftcap = 0.0
		case gemma4BLayerCount, gemma12BLayerCount, gemma27BLayerCount:
			// The 4B, 12B, and 27B models have rope scale unset
			// but it shuold be set to 8.0
			m.TextConfig.ropeScale = 8.0
		}
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

func (opts *TextConfig) ropeValuesForLayer(layer int) (base float32, scale float32) {
	if opts.slidingWindowPattern != nil && opts.slidingWindowPattern[layer] {
		return opts.ropeLocalBase, 1.0
	}

	// Standard Gemma3: only every n-th layer is global,
	// where n = gemmaGlobalCacheCount, otherwise use
	// the local rope base
	if (layer+1)%gemmaGlobalCacheCount > 0 {
		return opts.ropeLocalBase, 1.0
	}

	// default to global rope base
	return opts.ropeBase, opts.ropeScale
}

func (sa *TextSelfAttention) Forward(ctx ml.Context, layer int, hiddenState, positionIDs ml.Tensor, cache kvcache.Cache, opts *TextConfig) ml.Tensor {
	batchSize := hiddenState.Dim(1)

	ropeBase, ropeScale := opts.ropeValuesForLayer(layer)

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, opts.attnKeyLen, opts.numHeads, batchSize)
	q = sa.QueryNorm.Forward(ctx, q, opts.eps)
	q = opts.applyRotaryPositionEmbeddings(ctx, q, positionIDs, ropeBase, ropeScale)

	if opts.largeModelScaling {
		q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.hiddenSize/opts.numHeads)))
	} else {
		q = q.Scale(ctx, 1.0/math.Sqrt(float64(opts.attnKeyLen)))
	}

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, opts.attnKeyLen, opts.numKVHeads, batchSize)
	k = sa.KeyNorm.Forward(ctx, k, opts.eps)
	k = opts.applyRotaryPositionEmbeddings(ctx, k, positionIDs, ropeBase, ropeScale)

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, opts.attnValLen, opts.numKVHeads, batchSize)

	scaleFactor := 1.0
	kqv := nn.Attention(ctx, q, k, v, scaleFactor, cache)
	kqv = kqv.Reshape(ctx, opts.attnValLen*opts.numHeads, batchSize)

	return sa.Output.Forward(ctx, kqv)
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeBase, ropeScale := m.TextConfig.ropeValuesForLayer(layer)
	return m.applyRotaryPositionEmbeddings(ctx, key, shift, ropeBase, ropeScale), nil
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

	return m.OutputNorm.Forward(ctx, hiddenState, m.eps)
}
