package glmocr

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

type TextModelOptions struct {
	hiddenSize       int
	numHeads         int
	numKVHeads       int
	headDim          int
	rotaryDim        int
	intermediateSize int
	eps              float32
	ropeBase         float32
	mropeSections    []int
}

func (o *TextModelOptions) applyMRoPE(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	// With 4 sections for [temporal, height, width, unused]
	return nn.RoPE(ctx, states, positions, o.rotaryDim, o.ropeBase, 1.0, rope.WithMRoPE(o.mropeSections))
}

type TextSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_out"`
}

func (sa *TextSelfAttention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *TextModelOptions) ml.Tensor {
	batchSize := hiddenStates.Dim(1)

	// Separate Q, K, V projections
	q := sa.Query.Forward(ctx, hiddenStates)
	k := sa.Key.Forward(ctx, hiddenStates)
	v := sa.Value.Forward(ctx, hiddenStates)

	// Reshape for GQA
	q = q.Reshape(ctx, opts.headDim, opts.numHeads, batchSize)
	k = k.Reshape(ctx, opts.headDim, opts.numKVHeads, batchSize)
	v = v.Reshape(ctx, opts.headDim, opts.numKVHeads, batchSize)

	// Apply M-RoPE (multi-resolution rotary position embeddings)
	q = opts.applyMRoPE(ctx, q, positions)
	k = opts.applyMRoPE(ctx, k, positions)

	// Scaled dot-product attention with KV cache
	scaleFactor := 1.0 / math.Sqrt(float64(opts.headDim))
	kqv := nn.Attention(ctx, q, k, v, scaleFactor, cache)
	// Reshape attention output: [headDim, numHeads, batchSize] -> [numHeads*headDim, batchSize]
	// Note: numHeads * headDim = 16 * 128 = 2048, which is the attention hidden size
	kqv = kqv.Reshape(ctx, opts.numHeads*opts.headDim, batchSize)

	return sa.Output.Forward(ctx, kqv)
}

type TextMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *TextModelOptions) ml.Tensor {
	// SwiGLU: down(silu(gate(x)) * up(x))
	gate := mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, gate)
}

type TextDecoderLayer struct {
	// Input layernorm (before attention)
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *TextSelfAttention
	// Post self-attention layernorm (after attention, before residual add)
	PostAttnNorm *nn.RMSNorm `gguf:"post_attn_norm"`

	// FFN input layernorm (after first residual, before MLP)
	FFNNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP     *TextMLP
	// Post MLP layernorm (after MLP, before residual add)
	PostFFNNorm *nn.RMSNorm `gguf:"post_ffn_norm"`
}

func (l *TextDecoderLayer) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *TextModelOptions) ml.Tensor {
	// Attention block
	residual := hiddenStates
	hiddenStates = l.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = l.SelfAttention.Forward(ctx, hiddenStates, positions, cache, opts)
	hiddenStates = l.PostAttnNorm.Forward(ctx, hiddenStates, opts.eps)

	// Prune to output positions in final layer
	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)

	// MLP block
	residual = hiddenStates
	hiddenStates = l.FFNNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = l.MLP.Forward(ctx, hiddenStates, opts)
	hiddenStates = l.PostFFNNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = hiddenStates.Add(ctx, residual)

	return hiddenStates
}

type TextModel struct {
	TokenEmbedding *nn.Embedding      `gguf:"token_embd"`
	Layers         []TextDecoderLayer `gguf:"blk"`
	OutputNorm     *nn.RMSNorm        `gguf:"output_norm"`
	Output         *nn.Linear         `gguf:"output,alt:token_embd"`

	*TextModelOptions

	// positionCache stores the M-RoPE position for each token in the sequence.
	// This is needed because image tokens share the same base position but have
	// different height/width offsets, and the end token position depends on the
	// image grid dimensions.
	positionCache []int32
	ropeDelta     int32
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	// Clear position cache when KV cache shifts
	m.positionCache = nil
	m.ropeDelta = 0
	return m.applyMRoPE(ctx, key, shift), nil
}

func newTextModel(c fs.Config) *TextModel {
	hiddenSize := int(c.Uint("embedding_length", 1536))
	numHeads := int(c.Uint("attention.head_count", 16))
	numKVHeads := int(c.Uint("attention.head_count_kv", 8))
	intermediateSize := int(c.Uint("feed_forward_length", 4608))
	eps := c.Float("attention.layer_norm_rms_epsilon", 1e-5)
	ropeBase := c.Float("rope.freq_base", 10000)

	headDim := int(c.Uint("attention.key_length", uint32(hiddenSize/numHeads)))
	ropeDim := int(c.Uint("rope.dimension_count", uint32(headDim)))
	if ropeDim <= 0 {
		ropeDim = headDim
	}

	mropeSections := c.Ints("rope.mrope_section")
	var sectionInts []int

	if len(mropeSections) > 0 {
		sectionInts = make([]int, len(mropeSections))
		for i, section := range mropeSections {
			sectionInts[i] = int(section)
		}
	} else {
		// Default to GLM-OCR's HF ratio (2:3:3) scaled to rotaryDim/2.
		// For rotaryDim=64 this yields [8, 12, 12].
		total := ropeDim / 2
		if total <= 0 {
			total = 32
		}
		s0 := total * 2 / 8
		s1 := total * 3 / 8
		s2 := total - s0 - s1
		sectionInts = []int{s0, s1, s2}
	}

	// GGML rope_multi: sector = (dim_pair) % sum(sections), mapping each pair to its position dim
	rotaryDim := ropeDim

	return &TextModel{
		Layers: make([]TextDecoderLayer, c.Uint("block_count", 16)),
		TextModelOptions: &TextModelOptions{
			hiddenSize:       hiddenSize,
			numHeads:         numHeads,
			numKVHeads:       numKVHeads,
			headDim:          headDim,
			rotaryDim:        rotaryDim,
			intermediateSize: intermediateSize,
			eps:              eps,
			ropeBase:         ropeBase,
			mropeSections:    sectionInts,
		},
	}
}
