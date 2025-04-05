package llama4

import (
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
	Output      *nn.Linear `gguf:"attn_out"`
	RopeFactors ml.Tensor  `gguf:"rope_factors"`
}

func (sa *TextAttention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *TextModelOptions) ml.Tensor {
	batchSize, headDim := hiddenStates.Dim(1), opts.hiddenSize/opts.numHeads

	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, headDim, opts.numHeads, batchSize)
	key = key.Reshape(ctx, headDim, opts.numHeads, batchSize)
	value = value.Reshape(ctx, headDim, opts.numHeads, batchSize)

	query = query.RoPE(ctx, positions, sa.RopeFactors, uint32(opts.ropeDim), uint32(0), opts.ropeBase, opts.ropeScale)
	key = key.RoPE(ctx, positions, sa.RopeFactors, uint32(opts.ropeDim), uint32(0), opts.ropeBase, opts.ropeScale)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(headDim)), cache)
	attention = attention.Reshape(ctx, opts.hiddenSize, batchSize)
	return sa.Output.Forward(ctx, attention)
}

type TextMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *TextModelOptions) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type TextExperts struct {
	GateUp *nn.Linear `gguf:"gate_up_proj"`
	Down   *nn.Linear `gguf:"down_proj"`
}

func (e *TextExperts) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *TextModelOptions) ml.Tensor {
	// hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
	// gate_up = torch.matmul(hidden_states, self.gate_up_proj)
	// gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
	// next_states = torch.matmul((up * self.act_fn(gate)), self.down_proj)
	// next_states = next_states.view(-1, self.hidden_size)

	hiddenStates = hiddenStates.Reshape(ctx, opts.hiddenSize, hiddenStates.Dim(1)*hiddenStates.Dim(2), opts.numExperts)
	gateUp := e.GateUp.Forward(ctx, hiddenStates)

	gate := gateUp.View(ctx, 0, gateUp.Dim(0)/2, gateUp.Stride(1), gateUp.Dim(1), gateUp.Stride(2), gateUp.Dim(2), gateUp.Stride(3), gateUp.Dim(3))
	up := gateUp.View(ctx, gateUp.Stride(0)*gateUp.Dim(0)/2, gateUp.Dim(0)/2, gateUp.Stride(1), gateUp.Dim(1), gateUp.Stride(2), gateUp.Dim(2), gateUp.Stride(3), gateUp.Dim(3)).Contiguous(ctx)

	nextStates := e.Down.Forward(ctx, up.Mul(ctx, gate.SILU(ctx)))
	return nextStates.Reshape(ctx, opts.hiddenSize, nextStates.Dim(1)*nextStates.Dim(2))
}

type TextMOE struct {
	Router       *nn.Linear   `gguf:"router"`
	Experts      *TextExperts `gguf:"experts"`
	SharedExpert *TextMLP     `gguf:"shared_expert"`
}

func (moe *TextMOE) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *TextModelOptions) ml.Tensor {
	hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
	hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, sequenceLength*batchSize)
	routerLogits := moe.Router.Forward(ctx, hiddenStates)

	tokensPerExpert := batchSize * sequenceLength
	experts := routerLogits.TopK(ctx, opts.topK)

	scores := routerLogits.Reshape(ctx, 1, hiddenDim, tokensPerExpert).Rows(ctx, experts).Sigmoid(ctx)
	scores = moe.Experts.Forward(ctx, scores, opts)
	return moe.SharedExpert.Forward(ctx, scores, opts)
}

type TextMOELayer struct {
	AttentionNorm *nn.LayerNorm `gguf:"attn_norm"`
	Attention     *TextAttention

	FFNNorm *nn.LayerNorm `gguf:"ffn_norm"`
	MOE     *TextMOE
}

func (d *TextMOELayer) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *TextModelOptions) ml.Tensor {
	residual := hiddenStates

	// self attention
	hiddenStates = d.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = d.Attention.Forward(ctx, hiddenStates, positions, cache, opts)

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)
	residual = hiddenStates

	hiddenStates = d.FFNNorm.Forward(ctx, hiddenStates, opts.eps)
	return residual.Add(ctx, hiddenStates)
}

type TextMLPLayer struct {
	AttentionNorm *nn.LayerNorm `gguf:"attn_norm"`
	Attention     *TextAttention

	FFNNorm *nn.LayerNorm `gguf:"ffn_norm"`
	MLP     *TextMLP
}

func (d *TextMLPLayer) Forward(ctx ml.Context, hiddenStates, outputs ml.Tensor, cache kvcache.Cache, opts *TextModelOptions) ml.Tensor {
	residual := hiddenStates

	// self attention
	hiddenStates = d.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = d.Attention.Forward(ctx, hiddenStates, cache, opts)

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)
	residual = hiddenStates

	hiddenStates = d.FFNNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = d.MLP.Forward(ctx, hiddenStates, opts)

	return residual.Add(ctx, hiddenStates)
}

type TextLayer interface {
	Forward(ctx ml.Context, hiddenStates, outputs ml.Tensor, cache kvcache.Cache, opts *TextModelOptions) ml.Tensor
}

type TextModelOptions struct {
	hiddenSize, numHeads, numExperts int
	ropeDim                          int
	ropeBase, ropeScale              float32
	eps                              float32
	interleaveLayerStep              int
	topK                             int
}

type TextModel struct {
	Layers []TextLayer `gguf:"blk"`

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	OutputNorm     *nn.LayerNorm `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*TextModelOptions
}

func newTextModel(c fs.Config) *TextModel {
	layers := make([]TextLayer, c.Uint("block_count"))
	interleaveLayerStep := c.Uint("interleave_moe_layer_step", 1)
	for i := range layers {
		if (i+1)%int(interleaveLayerStep) == 0 {
			layers[i] = &TextMOELayer{}
		} else {
			layers[i] = &TextMLPLayer{}
		}
	}

	return &TextModel{
		Layers: layers,
		TextModelOptions: &TextModelOptions{
			hiddenSize:          int(c.Uint("embedding_length")),
			numHeads:            int(c.Uint("attention.head_count")),
			numExperts:          int(c.Uint("moe.expert_count")),
			ropeDim:             int(c.Uint("rope.dimension_count")),
			ropeBase:            c.Float("rope.freq_base"),
			ropeScale:           c.Float("rope.freq_scale", 1),
			eps:                 c.Float("attention.layer_norm_rms_epsilon"),
			interleaveLayerStep: int(c.Uint("interleave_moe_layer_step", 1)),
		},
	}
}

func (m *TextModel) Forward(ctx ml.Context, inputs, positions, outputs ml.Tensor, batch input.Batch, cache kvcache.Cache) ml.Tensor {
	hiddenStates := m.TokenEmbedding.Forward(ctx, inputs)

	for _, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, outputs, cache, m.TextModelOptions)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates)
}
