package qwen3

import (
	"cmp"
	"math"
	"strings"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Options struct {
	hiddenSize,
	numHeads,
	numKVHeads,
	keyLength,
	valueLength int

	eps,
	ropeBase,
	ropeScale float32
	ropeType              string
	originalContextLength int

	numExperts, numExpertsUsed int
	normTopKProb               bool
}

func (o Options) headDim() int {
	return cmp.Or(o.keyLength, o.valueLength, o.hiddenSize/o.numHeads)
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	opts := []func(*rope.Options){rope.WithTypeNeoX()}
	if o.ropeType == "yarn" {
		attnFactor := float32(1.0 / (1.0 + 0.1*math.Log(float64(o.ropeScale))))
		opts = append(opts,
			rope.WithOriginalContextLength(o.originalContextLength),
			rope.WithExtrapolationFactor(1.),
			rope.WithAttentionFactor(attnFactor),
		)
	}
	return nn.RoPE(ctx, states, positions, o.headDim(), o.ropeBase, 1./o.ropeScale, opts...)
}

type Attention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *Attention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenStates.Dim(1)

	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim(), opts.numHeads, batchSize)
	key = key.Reshape(ctx, opts.headDim(), opts.numKVHeads, batchSize)
	value = value.Reshape(ctx, opts.headDim(), opts.numKVHeads, batchSize)

	query = sa.QueryNorm.Forward(ctx, query, opts.eps)
	key = sa.KeyNorm.Forward(ctx, key, opts.eps)

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(opts.headDim())), cache)
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), batchSize)
	return sa.Output.Forward(ctx, attention)
}

type MLP interface {
	Forward(ml.Context, ml.Tensor, *Options) ml.Tensor
}

type sparse struct {
	Router *nn.Linear      `gguf:"ffn_gate_inp"`
	Gate   *nn.LinearBatch `gguf:"ffn_gate_exps"`
	Up     *nn.LinearBatch `gguf:"ffn_up_exps"`
	Down   *nn.LinearBatch `gguf:"ffn_down_exps"`
}

func (mlp *sparse) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
	hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, sequenceLength*batchSize)
	routerLogits := mlp.Router.Forward(ctx, hiddenStates)

	routingWeights := routerLogits.Softmax(ctx)
	selectedExperts := routingWeights.TopK(ctx, opts.numExpertsUsed)
	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, selectedExperts)
	if opts.normTopKProb {
		routingWeights = routingWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates.Dim(1))
		routingWeights = routingWeights.Div(ctx, routingWeights.SumRows(ctx))
		routingWeights = routingWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates.Dim(1))
	}

	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates, selectedExperts).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates, selectedExperts))

	experts := mlp.Down.Forward(ctx, hiddenStates, selectedExperts)
	experts = experts.Mul(ctx, routingWeights)

	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}

	return nextStates
}

type dense struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *dense) Forward(ctx ml.Context, hiddenStates ml.Tensor, _ *Options) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).
		SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	*Attention

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP
}

func (d *Layer) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenStates
	hiddenStates = d.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = d.Attention.Forward(ctx, hiddenStates, positions, cache, opts)

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = d.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = d.MLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}

type Model struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	Layers []Layer `gguf:"blk"`

	*Options
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	hiddenStates, err := m.forward(ctx, batch)
	if err != nil {
		return nil, err
	}

	return m.Output.Forward(ctx, hiddenStates), nil
}

// Forward implements model.Model.
func (m *Model) forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.Layers {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
		}

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Options)
	}

	return m.OutputNorm.Forward(ctx, hiddenStates, m.eps), nil
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.Options.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

var _ model.Model = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	layers := make([]Layer, c.Uint("block_count"))
	for i := range layers {
		if strings.HasSuffix(c.String("general.architecture"), "moe") {
			layers[i].MLP = &sparse{}
		} else {
			layers[i].MLP = &dense{}
		}
	}

	m := Model{
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
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
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		Layers: layers,
		Options: &Options{
			hiddenSize:            int(c.Uint("embedding_length")),
			numHeads:              int(c.Uint("attention.head_count")),
			numKVHeads:            int(c.Uint("attention.head_count_kv")),
			keyLength:             int(c.Uint("attention.key_length")),
			valueLength:           int(c.Uint("attention.value_length")),
			eps:                   c.Float("attention.layer_norm_rms_epsilon"),
			ropeType:              c.String("rope.scaling.type"),
			ropeBase:              c.Float("rope.freq_base"),
			ropeScale:             c.Float("rope.scaling.factor", 1),
			originalContextLength: int(c.Uint("rope.scaling.original_context_length")),
			numExperts:            int(c.Uint("expert_count")),
			numExpertsUsed:        int(c.Uint("expert_used_count")),
			normTopKProb:          c.Bool("norm_top_k_prob", true),
		},
	}

	m.Cache = kvcache.NewCausalCache(m.Shift)
	return &m, nil
}

func init() {
	model.Register("qwen3", New)
	model.Register("qwen3moe", New)
	model.Register("qwen3_embed", newEmbed)
}
