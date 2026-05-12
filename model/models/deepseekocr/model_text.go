package deepseekocr

import (
	"math"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

type textModel struct {
	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Blocks         []textBlock   `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output"`

	Options textOptions
}

func (m *textModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.Options.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

type textOptions struct {
	hiddenSize,
	numHeads,
	numKVHeads,
	numExperts,
	numExpertsUsed int
	ropeBase,
	ropeScale,
	eps float32
}

func (o textOptions) headDim() int {
	return o.hiddenSize / o.numHeads
}

func (o textOptions) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	return nn.RoPE(ctx, states, positions, o.headDim(), o.ropeBase, 1/o.ropeScale, rope.WithTypeNeoX())
}

type textBlock struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	Attention     *textAttention
	MLPNNorm      *nn.RMSNorm `gguf:"ffn_norm"`
	FeedForward   textFeedForward
}

func (m *textBlock) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts textOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = m.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = m.Attention.Forward(ctx, hiddenStates, positions, cache, opts)
	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = m.MLPNNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = m.FeedForward.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}

type textAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (m *textAttention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts textOptions) ml.Tensor {
	query := m.Query.Forward(ctx, hiddenStates)
	query = query.Reshape(ctx, opts.headDim(), opts.numHeads, -1)

	key := m.Key.Forward(ctx, hiddenStates)
	key = key.Reshape(ctx, opts.headDim(), opts.numKVHeads, -1)

	value := m.Value.Forward(ctx, hiddenStates)
	value = value.Reshape(ctx, opts.headDim(), opts.numKVHeads, -1)

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(opts.headDim())), cache)
	attention = attention.Reshape(ctx, -1, attention.Dim(2))
	return m.Output.Forward(ctx, attention)
}

type textFeedForward interface {
	Forward(ml.Context, ml.Tensor, textOptions) ml.Tensor
}

type textMoe struct {
	Router        *nn.Linear      `gguf:"ffn_gate_inp"`
	Gate          *nn.LinearBatch `gguf:"ffn_gate_exps"`
	Up            *nn.LinearBatch `gguf:"ffn_up_exps"`
	Down          *nn.LinearBatch `gguf:"ffn_down_exps"`
	SharedExperts *textMLP        `gguf:",suf:_shexp"`
}

func (m *textMoe) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts textOptions) ml.Tensor {
	scores := m.Router.Forward(ctx, hiddenStates).Softmax(ctx)
	indices := scores.TopK(ctx, opts.numExpertsUsed)
	weights := scores.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, indices)

	experts := hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))
	experts = m.Gate.Forward(ctx, experts, indices).SILU(ctx, m.Up.Forward(ctx, experts, indices))
	experts = m.Down.Forward(ctx, experts, indices)
	experts = experts.Mul(ctx, weights)

	expert := func(i int) ml.Tensor {
		return experts.View(
			ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2),
		)
	}

	routedStates := expert(0)
	for i := 1; i < opts.numExpertsUsed; i++ {
		routedStates = routedStates.Add(ctx, expert(i))
	}

	sharedStates := m.SharedExperts.Forward(ctx, hiddenStates, opts)
	return routedStates.Add(ctx, sharedStates)
}

type textMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (m *textMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, _ textOptions) ml.Tensor {
	hiddenStates = m.Gate.Forward(ctx, hiddenStates).SILU(ctx, m.Up.Forward(ctx, hiddenStates))
	return m.Down.Forward(ctx, hiddenStates)
}
