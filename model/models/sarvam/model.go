package sarvam

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

type Options struct {
	hiddenSize,
	numHeads,
	numKVHeads,
	headDim int

	eps,
	ropeBase float32

	numExperts,
	numExpertsUsed int
	normTopKProb        bool
	routedScalingFactor float32
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	return nn.RoPE(ctx, states, positions, o.headDim, o.ropeBase, 1.0, rope.WithTypeNeoX())
}

type Attention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	Key       *nn.Linear  `gguf:"attn_k"`
	Value     *nn.Linear  `gguf:"attn_v"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (attn *Attention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	seqLength := hiddenStates.Dim(1)

	query := attn.Query.Forward(ctx, hiddenStates)
	key := attn.Key.Forward(ctx, hiddenStates)
	value := attn.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim, opts.numHeads, seqLength)
	key = key.Reshape(ctx, opts.headDim, opts.numKVHeads, seqLength)
	value = value.Reshape(ctx, opts.headDim, opts.numKVHeads, seqLength)

	if attn.QueryNorm != nil {
		query = attn.QueryNorm.Forward(ctx, query, opts.eps)
	}
	if attn.KeyNorm != nil {
		key = attn.KeyNorm.Forward(ctx, key, opts.eps)
	}

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(opts.headDim)), cache)
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), seqLength)
	return attn.Output.Forward(ctx, attention)
}

type MLP interface {
	Forward(ml.Context, ml.Tensor, *Options) ml.Tensor
}

// sparse implements DeepSeek-style MoE with shared experts and sigmoid routing.
type sparse struct {
	Router       *nn.Linear `gguf:"ffn_gate_inp"`
	Gate         *nn.Linear `gguf:"ffn_gate_exps"`
	Up           *nn.Linear `gguf:"ffn_up_exps"`
	Down         *nn.Linear `gguf:"ffn_down_exps"`
	SharedExpert *dense     `gguf:",suf:_shexp"`
	ExpProbsBias ml.Tensor  `gguf:"exp_probs_b.bias,alt:exp_probs_b"`
}

func (moe *sparse) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	residuals := hiddenStates

	routerLogits := moe.Router.Forward(ctx, hiddenStates)
	scores := routerLogits.Sigmoid(ctx)

	if moe.ExpProbsBias != nil {
		scores = scores.Add(ctx, moe.ExpProbsBias)
	}

	topKIndices := scores.TopK(ctx, opts.numExpertsUsed)
	topKWeights := scores.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, topKIndices)

	if opts.normTopKProb {
		topKWeights = topKWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates.Dim(1))
		topKWeights = topKWeights.Div(ctx, topKWeights.SumRows(ctx))
		topKWeights = topKWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates.Dim(1))
	}

	topKWeights = topKWeights.Scale(ctx, float64(opts.routedScalingFactor))

	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

	upStates := moe.Up.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	hiddenStates = moe.Gate.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	hiddenStates = hiddenStates.SILU(ctx, upStates)

	experts := moe.Down.Weight.MulmatID(ctx, hiddenStates, topKIndices)
	experts = experts.Mul(ctx, topKWeights)

	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}

	if moe.SharedExpert != nil {
		sharedExpertResult := moe.SharedExpert.Forward(ctx, residuals, opts)
		nextStates = nextStates.Add(ctx, sharedExpertResult)
	}

	return nextStates
}

type dense struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *dense) Forward(ctx ml.Context, hiddenStates ml.Tensor, _ *Options) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	*Attention

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenStates
	hiddenStates = l.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = l.Attention.Forward(ctx, hiddenStates, positions, cache, opts)

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = l.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = l.MLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}

type Model struct {
	model.Base
	tokenizer.Tokenizer

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*Options
}

func New(c fs.Config) (model.Model, error) {
	layers := make([]Layer, c.Uint("block_count"))

	firstDenseLayerIndex := int(c.Uint("leading_dense_block_count"))
	for i := range layers {
		if i < firstDenseLayerIndex {
			layers[i].MLP = &dense{}
		} else {
			layers[i].MLP = &sparse{}
		}
	}

	var pre []string
	switch c.String("tokenizer.ggml.pre") {
	case "deepseek-v3":
		pre = []string{
			"\\p{N}{1,3}",
			`[一-龥぀-ゟ゠-ヿ]+`,
			"[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
		}
	default:
		return nil, model.ErrUnsupportedTokenizer
	}

	m := Model{
		Tokenizer: tokenizer.NewBytePairEncoding(
			&tokenizer.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			pre...,
		),
		Layers: layers,
		Options: &Options{
			hiddenSize:          int(c.Uint("embedding_length")),
			numHeads:            int(c.Uint("attention.head_count")),
			numKVHeads:          int(c.Uint("attention.head_count_kv")),
			headDim:             int(c.Uint("attention.key_length")),
			eps:                 c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:            c.Float("rope.freq_base"),
			numExperts:          int(c.Uint("expert_count")),
			numExpertsUsed:      int(c.Uint("expert_used_count")),
			normTopKProb:        c.Bool("expert_weights_norm", true),
			routedScalingFactor: c.Float("expert_weights_scale", 1),
		},
	}

	m.Cache = kvcache.NewCausalCache(m.Shift)
	return &m, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Options)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.Options.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

var _ model.Model = (*Model)(nil)

func init() {
	model.Register("sarvam_moe", New)
}
