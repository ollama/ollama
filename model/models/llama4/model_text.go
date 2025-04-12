package llama4

import (
	"cmp"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model/input"
)

type TextAttention struct {
	Query       *nn.Linear `gguf:"attn_q"`
	Key         *nn.Linear `gguf:"attn_k"`
	Value       *nn.Linear `gguf:"attn_v"`
	Output      *nn.Linear `gguf:"attn_output"`
	RopeFactors ml.Tensor  `gguf:"rope_factors"`
}

func (sa *TextAttention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, useRope bool, opts *TextOptions) ml.Tensor {
	batchSize, headDim := hiddenStates.Dim(1), cmp.Or(opts.headDim, opts.hiddenSize/opts.numHeads)

	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, headDim, opts.numHeads, batchSize)
	key = key.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	value = value.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	if useRope {
		query = query.RoPE(ctx, positions, sa.RopeFactors, uint32(opts.ropeDim), uint32(0), opts.ropeBase, opts.ropeScale)
		key = key.RoPE(ctx, positions, sa.RopeFactors, uint32(opts.ropeDim), uint32(0), opts.ropeBase, opts.ropeScale)

		if opts.useQKNorm {
			query = query.RMSNorm(ctx, nil, opts.eps)
			key = key.RMSNorm(ctx, nil, opts.eps)
		}
	}

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(headDim)), cache)
	attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)
	return sa.Output.Forward(ctx, attention)
}

type TextMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *TextOptions) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type TextExperts struct {
	Gate ml.Tensor `gguf:"ffn_gate_exps.weight"`
	Up   ml.Tensor `gguf:"ffn_up_exps.weight"`
	Down ml.Tensor `gguf:"ffn_down_exps.weight"`
}

func (e *TextExperts) Forward(ctx ml.Context, hiddenStates, routerLogits ml.Tensor, opts *TextOptions) ml.Tensor {
	experts := routerLogits.TopK(ctx, opts.numExpertsUsed)
	scores := routerLogits.Sigmoid(ctx).Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, experts)

	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))
	hiddenStates = hiddenStates.Repeat(ctx, 1, opts.numExpertsUsed)
	hiddenStates = hiddenStates.Mul(ctx, scores)

	upStates := e.Up.MulmatID(ctx, hiddenStates, experts)
	gateStates := e.Gate.MulmatID(ctx, hiddenStates, experts)
	downStates := e.Down.MulmatID(ctx, upStates.Mul(ctx, gateStates.SILU(ctx)), experts)

	nextStates := downStates.View(ctx, 0, hiddenStates.Dim(0), downStates.Stride(2), hiddenStates.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates.Add(ctx, downStates.View(ctx, i*downStates.Stride(1), hiddenStates.Dim(0), downStates.Stride(2), hiddenStates.Dim(2)))
	}

	return nextStates
}

// TextSharedExpert is TextMLP with different names
type TextSharedExpert struct {
	Gate *nn.Linear `gguf:"ffn_gate_shexp"`
	Up   *nn.Linear `gguf:"ffn_up_shexp"`
	Down *nn.Linear `gguf:"ffn_down_shexp"`
}

func (mlp *TextSharedExpert) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *TextOptions) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type TextMOE struct {
	Router       *nn.Linear `gguf:"ffn_gate_inp"`
	Experts      *TextExperts
	SharedExpert *TextSharedExpert
}

func (moe *TextMOE) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *TextOptions) ml.Tensor {
	hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
	hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, sequenceLength*batchSize)
	routerLogits := moe.Router.Forward(ctx, hiddenStates)

	sharedStates := moe.SharedExpert.Forward(ctx, hiddenStates, opts)
	routedStates := moe.Experts.Forward(ctx, hiddenStates, routerLogits, opts)
	return sharedStates.Add(ctx, routedStates)
}

type TextFeedForward interface {
	Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *TextOptions) ml.Tensor
}

type TextLayer struct {
	AttentionNorm *nn.LayerNorm `gguf:"attn_norm"`
	Attention     *TextAttention

	FFNNorm     *nn.LayerNorm `gguf:"ffn_norm"`
	FeedForward TextFeedForward
}

func (d *TextLayer) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, useRope bool, opts *TextOptions) ml.Tensor {
	residual := hiddenStates

	// self attention
	hiddenStates = d.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = d.Attention.Forward(ctx, hiddenStates, positions, cache, useRope, opts)

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)
	residual = hiddenStates

	hiddenStates = d.FFNNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = d.FeedForward.Forward(ctx, hiddenStates, opts)

	return residual.Add(ctx, hiddenStates)
}

type TextOptions struct {
	hiddenSize                    int
	numHeads, numKVHeads, headDim int
	numExperts, numExpertsUsed    int
	ropeDim                       int
	ropeBase, ropeScale           float32
	eps                           float32
	interleaveLayerStep           int
	topK                          int
	useQKNorm                     bool
}

type TextModel struct {
	Layers []TextLayer `gguf:"blk"`

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	OutputNorm     *nn.LayerNorm `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*TextOptions
}

func newTextModel(c fs.Config) *TextModel {
	layers := make([]TextLayer, c.Uint("block_count"))
	interleaveLayerStep := c.Uint("interleave_moe_layer_step", 1)
	for i := range layers {
		if (i+1)%int(interleaveLayerStep) == 0 {
			layers[i] = TextLayer{FeedForward: &TextMOE{}}
		} else {
			layers[i] = TextLayer{FeedForward: &TextMLP{}}
		}
	}

	return &TextModel{
		Layers: layers,
		TextOptions: &TextOptions{
			hiddenSize:          int(c.Uint("embedding_length")),
			numHeads:            int(c.Uint("attention.head_count")),
			numKVHeads:          int(c.Uint("attention.head_count_kv")),
			headDim:             int(c.Uint("attention.head_dim", 128)),
			numExperts:          int(c.Uint("expert_count")),
			numExpertsUsed:      int(c.Uint("expert_used_count")),
			ropeDim:             int(c.Uint("rope.dimension_count")),
			ropeBase:            c.Float("rope.freq_base"),
			ropeScale:           c.Float("rope.freq_scale", 1),
			eps:                 c.Float("attention.layer_norm_rms_epsilon"),
			interleaveLayerStep: int(c.Uint("interleave_moe_layer_step", 1)),
			useQKNorm:           c.Bool("use_qk_norm", true),
		},
	}
}

func (m *TextModel) Forward(ctx ml.Context, inputs, positions, outputs ml.Tensor, batch input.Batch, cache kvcache.Cache) ml.Tensor {
	hiddenStates := m.TokenEmbedding.Forward(ctx, inputs)

	for i, layer := range m.Layers {
		cache.SetLayer(i)
		wc := cache.(*kvcache.WrapperCache)
		wc.SetLayerType(1)
		useChunkedAttention := (i+1)%4 != 0
		if useChunkedAttention {
			wc.SetLayerType(0)
		}

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, lastLayerOutputs, cache, useChunkedAttention, m.TextOptions)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates)
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return key.RoPE(ctx, shift, m.Layers[layer].Attention.RopeFactors, uint32(0), uint32(m.ropeDim), m.ropeBase, m.ropeScale), nil
}
