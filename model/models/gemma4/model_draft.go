package gemma4

import (
	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

type DraftModel struct {
	PreProjection  *nn.Linear    `gguf:"pre_projection"`
	PostProjection *nn.Linear    `gguf:"post_projection"`
	EmbedTokens    *nn.Embedding `gguf:"token_embd"`
	Layers         []DraftLayer  `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Centroids      *nn.Linear    `gguf:"centroids"`
	TokenOrdering  ml.Tensor     `gguf:"token_ordering.weight"`
	DraftOptions
}

type DraftOptions struct {
	hiddenSize           int
	numHeads             int
	headDim              int
	globalHeadDim        int
	eps                  float32
	ropeBase             float32
	ropeLocalBase        float32
	partialRotaryDims    int
	slidingWindowPattern []bool
	useOrderedEmbeddings bool
	numCentroids         int
	centroidTopK         int
	vocabSize            int
}

type DraftLayer struct {
	AttentionNorm     *nn.RMSNorm `gguf:"attn_norm"`
	PostAttentionNorm *nn.RMSNorm `gguf:"post_attention_norm"`
	MLPNorm           *nn.RMSNorm `gguf:"ffn_norm"`
	PostMLPNorm       *nn.RMSNorm `gguf:"post_ffw_norm"`
	Attention         *DraftAttention
	MLP               *TextMLP
	LayerScalar       ml.Tensor `gguf:"layer_output_scale.weight"`
	isSliding         bool
}

type DraftAttention struct {
	QProj     *nn.Linear  `gguf:"attn_q"`
	OProj     *nn.Linear  `gguf:"attn_output"`
	QNorm     *nn.RMSNorm `gguf:"attn_q_norm"`
	RopeFreqs ml.Tensor   `gguf:"rope_freqs.weight"`
}

func newDraftModel(c fs.Config) *DraftModel {
	numLayers := int(c.Uint("draft.block_count", 0))
	if numLayers == 0 {
		return nil
	}

	headDim := int(c.Uint("draft.attention.key_length_swa",
		c.Uint("attention.key_length_swa", 256)))
	globalHeadDim := int(c.Uint("draft.attention.key_length",
		c.Uint("attention.key_length", 512)))

	partialRotaryDims := int(c.Uint("draft.rope.dimension_count", 0))
	if partialRotaryDims == 0 {
		partialFactor := c.Float("draft.rope.partial_rotary_factor",
			c.Float("rope.partial_rotary_factor", 1.0))
		partialRotaryDims = int(float32(globalHeadDim) * partialFactor)
	}

	ropeBase := c.Float("draft.rope.freq_base",
		c.Float("rope.freq_base", 1000000.0))
	ropeLocalBase := c.Float("draft.rope.freq_base_swa",
		c.Float("rope.freq_base_swa",
			c.Float("rope.local.freq_base", 10000.0)))

	slidingPattern := c.Bools("draft.attention.sliding_window_pattern")
	if len(slidingPattern) == 0 {
		slidingPattern = c.Bools("attention.sliding_window_pattern")
		if len(slidingPattern) > numLayers {
			slidingPattern = slidingPattern[:numLayers]
		}
	}

	m := &DraftModel{
		Layers: make([]DraftLayer, numLayers),
		DraftOptions: DraftOptions{
			hiddenSize:           int(c.Uint("draft.embedding_length", c.Uint("embedding_length"))),
			numHeads:             int(c.Uint("draft.attention.head_count", c.Uint("attention.head_count"))),
			headDim:              headDim,
			globalHeadDim:        globalHeadDim,
			eps:                  c.Float("draft.attention.layer_norm_rms_epsilon", c.Float("attention.layer_norm_rms_epsilon", 1e-06)),
			ropeBase:             ropeBase,
			ropeLocalBase:        ropeLocalBase,
			partialRotaryDims:    partialRotaryDims,
			slidingWindowPattern: slidingPattern,
			useOrderedEmbeddings: c.Bool("draft.use_ordered_embeddings", false),
			numCentroids:         int(c.Uint("draft.num_centroids", 2048)),
			centroidTopK:         int(c.Uint("draft.centroid_intermediate_top_k", 32)),
			vocabSize:            int(c.Uint("tokenizer.ggml.vocab_size", c.Uint("vocab_size", 262144))),
		},
	}

	for i := range m.Layers {
		m.Layers[i].isSliding = m.isLayerSliding(i)
	}

	return m
}

func (o *DraftOptions) isLayerSliding(layer int) bool {
	if layer < len(o.slidingWindowPattern) {
		return o.slidingWindowPattern[layer]
	}
	return false
}

func (o *DraftOptions) ropeForLayer(layer int) (base float32, dims int) {
	if o.isLayerSliding(layer) {
		return o.ropeLocalBase, o.headDim
	}
	return o.ropeBase, o.partialRotaryDims
}

func (o *DraftOptions) headDimForLayer(layer int) int {
	if o.isLayerSliding(layer) {
		return o.headDim
	}
	return o.globalHeadDim
}

// Draft runs the assistant model on concatenated [embedding, hidden] input at a
// fixed position, reusing the target model's KV cache. Returns next-token logits
// and the projected hidden state for chaining to the next draft iteration.
func (m *DraftModel) Draft(ctx ml.Context, inputEmbeds ml.Tensor, position int32, cache kvcache.Cache, targetOpts *TextOptions) (logits, projectedHidden ml.Tensor) {
	batchSize := inputEmbeds.Dim(1)
	positions := ctx.Input().FromInts([]int32{position}, 1)

	h := m.PreProjection.Forward(ctx, inputEmbeds)

	for i, layer := range m.Layers {
		targetLayer := m.mapToTargetLayer(i, targetOpts)
		cache.SetLayer(targetLayer)
		if wc, ok := cache.(*kvcache.WrapperCache); ok {
			if m.isLayerSliding(i) {
				wc.SetLayerType(cacheTypeSWA)
			} else {
				wc.SetLayerType(cacheTypeCausal)
			}
		}

		h = layer.Forward(ctx, i, h, positions, batchSize, cache, &m.DraftOptions)
	}

	hidden := m.OutputNorm.Forward(ctx, h, m.eps)
	projectedHidden = m.PostProjection.Forward(ctx, hidden)
	logits = m.unembed(ctx, hidden)
	return logits, projectedHidden
}

// mapToTargetLayer finds the target model layer whose KV cache the draft layer
// should read from. The assistant's layers are much fewer than the target's, so
// we map each draft layer to the last target layer of the same attention type
// (sliding or full) before the KV-shared region.
func (m *DraftModel) mapToTargetLayer(draftLayer int, targetOpts *TextOptions) int {
	isSliding := m.isLayerSliding(draftLayer)

	// Walk backwards through target layers to find the last non-shared layer
	// of the matching type.
	for i := targetOpts.hiddenLayers - 1; i >= 0; i-- {
		if _, isShared := targetOpts.kvDonorMap[i]; isShared {
			continue
		}
		if targetOpts.isLocal(i) == isSliding {
			return i
		}
	}
	return 0
}

func (l *DraftLayer) Forward(ctx ml.Context, layer int, hiddenState, positions ml.Tensor, batchSize int, cache kvcache.Cache, opts *DraftOptions) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.Attention.Forward(ctx, layer, hiddenState, positions, batchSize, l.isSliding, cache, opts)
	hiddenState = l.PostAttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState)
	hiddenState = l.PostMLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = hiddenState.Add(ctx, residual)

	if l.LayerScalar != nil {
		hiddenState = hiddenState.Mul(ctx, l.LayerScalar)
	}

	return hiddenState
}

func (a *DraftAttention) Forward(ctx ml.Context, layer int, hiddenState, positions ml.Tensor, batchSize int, isSliding bool, cache kvcache.Cache, opts *DraftOptions) ml.Tensor {
	hd := opts.headDimForLayer(layer)
	ropeBase, ropeDims := opts.ropeForLayer(layer)

	q := a.QProj.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, hd, opts.numHeads, batchSize)
	q = a.QNorm.Forward(ctx, q, opts.eps)

	ropeOpts := []func(*rope.Options){rope.WithTypeNeoX()}
	if a.RopeFreqs != nil && !isSliding {
		ropeOpts = append(ropeOpts, rope.WithFactors(a.RopeFreqs))
	}
	q = nn.RoPE(ctx, q, positions, ropeDims, ropeBase, 1.0, ropeOpts...)

	attention := nn.Attention(ctx, q, nil, nil, 1.0, cache)

	attention = attention.Reshape(ctx, hd*opts.numHeads, batchSize)
	return a.OProj.Forward(ctx, attention)
}

func (m *DraftModel) unembed(ctx ml.Context, hidden ml.Tensor) ml.Tensor {
	if !m.useOrderedEmbeddings || m.Centroids == nil || m.TokenOrdering == nil {
		return m.EmbedTokens.Weight.Mulmat(ctx, hidden)
	}
	return m.centroidMaskedUnembed(ctx, hidden)
}

func (m *DraftModel) centroidMaskedUnembed(ctx ml.Context, hidden ml.Tensor) ml.Tensor {
	vocabSize := m.vocabSize
	numCentroids := m.numCentroids
	topK := m.centroidTopK
	vocabPerCentroid := vocabSize / numCentroids

	centroidLogits := m.Centroids.Forward(ctx, hidden)

	// TopK operates on 1D; negate for top-K via smallest-K
	negLogits := centroidLogits.Scale(ctx, -1.0)
	topKIndices := negLogits.TopK(ctx, topK)

	ordering := m.TokenOrdering.Reshape(ctx, vocabPerCentroid, numCentroids)
	selectedCanonical := ordering.Rows(ctx, topKIndices)

	hiddenSize := hidden.Dim(0)
	batchSize := hidden.Dim(1)
	numSelected := topK * vocabPerCentroid

	selectedFlat := selectedCanonical.Reshape(ctx, batchSize*numSelected)
	embeddings := m.EmbedTokens.Forward(ctx, selectedFlat)
	embeddings = embeddings.Reshape(ctx, hiddenSize, numSelected, batchSize)

	hiddenExpanded := hidden.Reshape(ctx, hiddenSize, 1, batchSize)
	selectedLogits := embeddings.MulmatFullPrec(ctx, hiddenExpanded)
	selectedLogits = selectedLogits.Reshape(ctx, numSelected, batchSize)

	// Scatter into full vocab tensor initialized to -inf
	fullLogits := ctx.Zeros(ml.DTypeF32, vocabSize, batchSize)
	fullLogits = fullLogits.Add(ctx, ctx.FromFloats([]float32{-1e30}))

	canonicalIndices := selectedCanonical.Reshape(ctx, numSelected, batchSize)
	fullLogits = fullLogits.SetRows(ctx, selectedLogits, canonicalIndices)

	return fullLogits
}
