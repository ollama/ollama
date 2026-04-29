package laguna

import (
	"fmt"
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

const (
	cacheTypeSWA = iota
	cacheTypeCausal
)

type Options struct {
	hiddenSize int
	headDim    int

	numHeads   []int
	numKVHeads int

	eps float32

	slidingWindow        int
	slidingWindowPattern []bool

	fullRopeDim                   int
	fullRopeBase, fullRopeScale   float32
	fullRopeOriginalContextLength int
	fullRopeAttentionFactor       float32
	fullRopeBetaFast              float32
	fullRopeBetaSlow              float32

	swaRopeDim                 int
	swaRopeBase, swaRopeScale  float32
	numExperts, numExpertsUsed int
	normTopKProb               bool
	routedScalingFactor        float32
	decoderSparseStep          int
	denseLayers                map[int]bool
}

func (o *Options) numHeadsForLayer(layer int) int {
	if layer < len(o.numHeads) && o.numHeads[layer] > 0 {
		return o.numHeads[layer]
	}
	if len(o.numHeads) > 0 && o.numHeads[0] > 0 {
		return o.numHeads[0]
	}
	return 1
}

func (o *Options) layerIsSliding(layer int) bool {
	return layer < len(o.slidingWindowPattern) && o.slidingWindowPattern[layer]
}

func (o *Options) layerUsesMoE(layer int) bool {
	if o.numExperts == 0 || o.denseLayers[layer] {
		return false
	}
	step := o.decoderSparseStep
	if step <= 0 {
		step = 1
	}
	return (layer+1)%step == 0
}

func (o *Options) applyRotaryPositionEmbeddings(ctx ml.Context, layer int, states, positions ml.Tensor) ml.Tensor {
	opts := []func(*rope.Options){rope.WithTypeNeoX()}
	if o.layerIsSliding(layer) {
		return nn.RoPE(ctx, states, positions, o.swaRopeDim, o.swaRopeBase, 1./o.swaRopeScale, opts...)
	}

	opts = append(opts,
		rope.WithOriginalContextLength(o.fullRopeOriginalContextLength),
		rope.WithExtrapolationFactor(1),
		rope.WithAttentionFactor(o.fullRopeAttentionFactor),
		rope.WithBetaFast(o.fullRopeBetaFast),
		rope.WithBetaSlow(o.fullRopeBetaSlow),
	)
	return nn.RoPE(ctx, states, positions, o.fullRopeDim, o.fullRopeBase, 1./o.fullRopeScale, opts...)
}

type Attention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Gate      *nn.Linear  `gguf:"attn_g"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *Attention) Forward(ctx ml.Context, layer int, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenStates.Dim(1)
	numHeads := opts.numHeadsForLayer(layer)

	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)
	gate := sa.Gate.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim, numHeads, batchSize)
	key = key.Reshape(ctx, opts.headDim, opts.numKVHeads, batchSize)
	value = value.Reshape(ctx, opts.headDim, opts.numKVHeads, batchSize)

	query = sa.QueryNorm.Forward(ctx, query, opts.eps)
	key = sa.KeyNorm.Forward(ctx, key, opts.eps)

	query = opts.applyRotaryPositionEmbeddings(ctx, layer, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, layer, key, positions)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(opts.headDim)), cache)
	// Laguna applies the per-head gate softplus in float32, then casts back.
	gate = gate.Cast(ctx, ml.DTypeF32).Softplus(ctx).Cast(ctx, attention.DType())
	attention = attention.Mul(ctx, gate.Reshape(ctx, 1, numHeads, batchSize))
	attention = attention.Reshape(ctx, opts.headDim*numHeads, batchSize)
	return sa.Output.Forward(ctx, attention)
}

type MLP interface {
	Forward(ml.Context, ml.Tensor, *Options) ml.Tensor
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

type sparse struct {
	Router       *nn.Linear      `gguf:"ffn_gate_inp"`
	Gate         *nn.LinearBatch `gguf:"ffn_gate_exps"`
	Up           *nn.LinearBatch `gguf:"ffn_up_exps"`
	Down         *nn.LinearBatch `gguf:"ffn_down_exps"`
	SharedExpert *dense          `gguf:",suf:_shexp"`
	ExpProbsBias ml.Tensor       `gguf:"exp_probs_b.bias,alt:exp_probs_b"`
}

func (moe *sparse) topKIndices(ctx ml.Context, scores ml.Tensor, opts *Options) ml.Tensor {
	if moe.ExpProbsBias != nil {
		scores = scores.Add(ctx, moe.ExpProbsBias)
	}
	return scores.TopK(ctx, opts.numExpertsUsed)
}

func (moe *sparse) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	residual := hiddenStates

	scores := moe.Router.Forward(ctx, hiddenStates).Cast(ctx, ml.DTypeF32).Sigmoid(ctx)
	selectedExperts := moe.topKIndices(ctx, scores, opts)
	routingWeights := scores.Reshape(ctx, 1, opts.numExperts, hiddenStates.Dim(1)).Rows(ctx, selectedExperts)
	if opts.normTopKProb {
		routingWeights = routingWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates.Dim(1))
		routingWeights = routingWeights.Div(ctx, routingWeights.SumRows(ctx))
		routingWeights = routingWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates.Dim(1))
	}
	routingWeights = routingWeights.Scale(ctx, float64(opts.routedScalingFactor))

	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))
	upStates := moe.Up.Forward(ctx, hiddenStates, selectedExperts)
	hiddenStates = moe.Gate.Forward(ctx, hiddenStates, selectedExperts).SILU(ctx, upStates)

	experts := moe.Down.Forward(ctx, hiddenStates, selectedExperts)
	experts = experts.Mul(ctx, routingWeights)

	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}

	return nextStates.Add(ctx, moe.SharedExpert.Forward(ctx, residual, opts))
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	*Attention

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP
}

func (l *Layer) Forward(ctx ml.Context, layer int, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenStates
	hiddenStates = l.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = l.Attention.Forward(ctx, layer, hiddenStates, positions, cache, opts)

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
	if c.Bool("attention.sink_enabled") {
		return nil, fmt.Errorf("laguna: SWA attention sinks are not supported")
	}
	if c.Uint("attention.gating_type") != 1 {
		return nil, fmt.Errorf("laguna: unsupported attention gating type %d", c.Uint("attention.gating_type"))
	}
	if !c.Bool("attention.qk_norm") {
		return nil, fmt.Errorf("laguna: Q/K RMSNorm is required")
	}
	if gating := c.Uint("expert_gating_func"); gating != 2 {
		return nil, fmt.Errorf("laguna: unsupported expert gating function %d", gating)
	}

	numLayers := int(c.Uint("block_count"))
	opts := newOptions(c, numLayers)
	layers := make([]Layer, numLayers)
	for i := range layers {
		if opts.layerUsesMoE(i) {
			layers[i].MLP = &sparse{}
		} else {
			layers[i].MLP = &dense{}
		}
	}

	var pre []string
	switch c.String("tokenizer.ggml.pre") {
	case "laguna":
		pre = []string{
			`(?:\r?\n)+(?!\r?\n)`,
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
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
		Layers:  layers,
		Options: opts,
	}

	m.Cache = kvcache.NewWrapperCache(
		kvcache.NewSWACache(int32(opts.slidingWindow), m.Shift),
		kvcache.NewCausalCache(m.Shift),
	)
	return &m, nil
}

func newOptions(c fs.Config, numLayers int) *Options {
	denseLayers := make(map[int]bool)
	for _, layer := range configUints(c, "dense_layers") {
		denseLayers[int(layer)] = true
	}
	for i := range c.Uint("leading_dense_block_count") {
		denseLayers[int(i)] = true
	}

	fullRopeScale := c.Float("rope.scaling.factor", 1)
	if fullRopeScale == 0 {
		fullRopeScale = 1
	}
	swaRopeScale := c.Float("rope.swa.scaling.factor", 1)
	if swaRopeScale == 0 {
		swaRopeScale = 1
	}
	fullRopeType := c.String("rope.scaling.type")
	fullRopeAttentionFactor := lagunaAttentionFactor(fullRopeType, fullRopeScale, c.Float("rope.scaling.attn_factor"))

	return &Options{
		hiddenSize:                    int(c.Uint("embedding_length")),
		headDim:                       int(c.Uint("attention.key_length")),
		numHeads:                      expandIntArray(configUints(c, "attention.head_count"), numLayers, c.Uint("attention.head_count", 1)),
		numKVHeads:                    int(c.Uint("attention.head_count_kv")),
		eps:                           c.Float("attention.layer_norm_rms_epsilon", 1e-6),
		slidingWindow:                 int(c.Uint("attention.sliding_window", 512)),
		slidingWindowPattern:          slidingWindowPattern(c, numLayers),
		fullRopeDim:                   int(c.Uint("rope.dimension_count", c.Uint("attention.key_length"))),
		fullRopeBase:                  c.Float("rope.freq_base", 500000),
		fullRopeScale:                 fullRopeScale,
		fullRopeOriginalContextLength: int(c.Uint("rope.scaling.original_context_length", 4096)),
		fullRopeAttentionFactor:       fullRopeAttentionFactor,
		fullRopeBetaFast:              c.Float("rope.scaling.beta_fast", 64),
		fullRopeBetaSlow:              c.Float("rope.scaling.beta_slow", 1),
		swaRopeDim:                    int(c.Uint("rope.swa.dimension_count", c.Uint("attention.key_length"))),
		swaRopeBase:                   c.Float("rope.swa.freq_base", 10000),
		swaRopeScale:                  swaRopeScale,
		numExperts:                    int(c.Uint("expert_count")),
		numExpertsUsed:                int(c.Uint("expert_used_count")),
		normTopKProb:                  c.Bool("expert_weights_norm", true),
		routedScalingFactor:           c.Float("expert_weights_scale", 1),
		decoderSparseStep:             int(c.Uint("decoder_sparse_step", 1)),
		denseLayers:                   denseLayers,
	}
}

func lagunaAttentionFactor(ropeType string, scaleFactor, attentionFactor float32) float32 {
	if attentionFactor != 0 {
		return attentionFactor
	}
	if ropeType == "yarn" && scaleFactor > 1 {
		return float32(0.1*math.Log(float64(scaleFactor)) + 1)
	}
	return 1
}

func slidingWindowPattern(c fs.Config, numLayers int) []bool {
	pattern := c.Bools("attention.sliding_window_pattern")
	if len(pattern) == numLayers {
		return pattern
	}

	layerTypes := configUints(c, "attention.layer_types")
	if len(layerTypes) == numLayers {
		pattern = make([]bool, numLayers)
		for i, layerType := range layerTypes {
			pattern[i] = layerType == 1
		}
		return pattern
	}

	return make([]bool, numLayers)
}

func configUints(c fs.Config, key string) []uint32 {
	keyExists := c.Value(c.Architecture()+"."+key) != nil || c.Value(key) != nil
	if cc, ok := c.(interface {
		Uints(string, ...[]uint32) []uint32
	}); ok {
		if values := cc.Uints(key); len(values) > 0 && (keyExists || !(len(values) == 1 && values[0] == 0)) {
			return values
		}
	}

	ints := c.Ints(key)
	if len(ints) > 0 && (keyExists || !(len(ints) == 1 && ints[0] == 0)) {
		values := make([]uint32, len(ints))
		for i, v := range ints {
			values[i] = uint32(v)
		}
		return values
	}

	if scalar := c.Uint(key); scalar != 0 {
		return []uint32{scalar}
	}
	return nil
}

func expandIntArray(values []uint32, n int, fallback uint32) []int {
	if len(values) == 0 {
		values = []uint32{fallback}
	}
	defaultValue := values[0]
	if len(values) == 1 {
		defaultValue = values[0]
	}

	out := make([]int, n)
	for i := range out {
		if i < len(values) {
			out[i] = int(values[i])
		} else {
			out[i] = int(defaultValue)
		}
	}
	return out
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.Options.applyRotaryPositionEmbeddings(ctx, layer, key, shift), nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.Layers {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
			if wrapper, ok := m.Cache.(*kvcache.WrapperCache); ok {
				cacheType := cacheTypeCausal
				if m.Options.layerIsSliding(i) {
					cacheType = cacheTypeSWA
				}
				wrapper.SetLayerType(cacheType)
			}
		}

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, i, hiddenStates, positions, outputs, m.Cache, m.Options)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("laguna", New)
}

var _ model.Model = (*Model)(nil)
