package nemotronh

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

// Options contains model configuration
type Options struct {
	hiddenSize int
	numHeads   int // attention heads
	numKVHeads int // KV heads for attention layers
	headDim    int
	eps        float32

	// Mamba2 SSM config
	ssmDConv  int // conv kernel size
	ssmDInner int // inner dimension (d_inner)
	ssmDState int // state dimension
	ssmNHead  int // number of SSM heads (dt_rank)
	ssmNGroup int // number of groups for B, C

	// Per-layer configuration
	isRecurrent []bool // true = Mamba2, false = attention or FFN
	nFF         []int  // n_ff per layer (0 = attention-only)

	// Attention scale
	attentionScale float64

	// MoE config
	numExperts            int
	numExpertsUsed        int
	expertWeightsNorm     bool
	expertWeightsScale    float32
	expertWeightsNormClip float32
}

func (o Options) getHeadDim() int {
	if o.headDim > 0 {
		return o.headDim
	}
	if o.numHeads <= 0 {
		return 0
	}
	return o.hiddenSize / o.numHeads
}

// Operator is the interface for layer operators (Mamba2 or Attention)
type Operator interface {
	Forward(ctx ml.Context, hiddenStates ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error)
}

// MLP is the interface for feedforward networks
type MLP interface {
	Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor
}

// Layer represents a single transformer layer
type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	Operator      Operator    // Mamba2, Attention, or nil (for FFN-only layers)
	MLP           MLP         // Dense or MoE FFN, or nil
}

func (l *Layer) Forward(ctx ml.Context, layer int, hiddenStates, outputs ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error) {
	residual := hiddenStates

	// Pre-layer norm
	hiddenStates = l.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)

	// Layer operator (Mamba2, Attention, or FFN)
	if l.Operator != nil {
		var err error
		hiddenStates, err = l.Operator.Forward(ctx, hiddenStates, cache, opts)
		if err != nil {
			return nil, err
		}
	} else if l.MLP != nil {
		// FFN-only layer
		hiddenStates = l.MLP.Forward(ctx, hiddenStates, opts)
	}

	// Output projection for last layer
	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	// Residual connection
	return hiddenStates.Add(ctx, residual), nil
}

// Model is the main Nemotron-H model
type Model struct {
	model.Base
	tokenizer.Tokenizer

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	Layers []Layer `gguf:"blk"`

	*Options
}

// Shift is used for KV cache position shifting.
// Nemotron-H attention does not apply RoPE, so keys do not need to be transformed.
func Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return key, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	cache := m.Cache.(*HybridCache)

	for i, layer := range m.Layers {
		cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		var err error
		hiddenStates, err = layer.Forward(ctx, i, hiddenStates, outputs, cache, m.Options)
		if err != nil {
			return nil, err
		}
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func New(c fs.Config) (model.Model, error) {
	numLayers := int(c.Uint("block_count"))
	layers := make([]Layer, numLayers)

	// Get per-layer configuration from GGUF metadata
	// Use the same interface pattern as qwen3next
	type perLayerConfig interface {
		HeadCount() []uint64
		HeadCountKV() []uint64
		FFNLength() []uint64
	}

	var headCount []uint64
	var headCountKV []uint64
	var ffnLength []uint64

	if plc, ok := c.(perLayerConfig); ok {
		headCount = plc.HeadCount()
		headCountKV = plc.HeadCountKV()
		ffnLength = plc.FFNLength()
	}

	// Build per-layer arrays with defaults
	isRecurrent := make([]bool, numLayers)
	nFF := make([]int, numLayers)

	for i := range numLayers {
		// Get per-layer values
		kvHeads := uint64(1) // Default non-zero
		if i < len(headCountKV) {
			kvHeads = headCountKV[i]
		}
		ff := uint64(0)
		if i < len(ffnLength) {
			ff = ffnLength[i]
		}
		nFF[i] = int(ff)

		// A layer is recurrent IFF n_head_kv == 0 AND n_ff == 0
		// This matches llama.cpp behavior for Nemotron-H
		isRecurrent[i] = kvHeads == 0 && ff == 0
	}

	// Determine if MoE
	isMoE := c.Uint("expert_count") > 0

	for i := range layers {
		if isRecurrent[i] {
			// Mamba2 layer
			layers[i].Operator = &Mamba2{Layer: i}
		} else if nFF[i] == 0 {
			// Attention-only layer (n_head_kv > 0, n_ff == 0)
			layers[i].Operator = &Attention{}
		} else {
			// FFN layer (n_ff > 0)
			if isMoE {
				layers[i].MLP = &MoESparse{}
			} else {
				layers[i].MLP = &Dense{}
			}
		}
	}

	// Get attention head configuration
	numHeads := int(c.Uint("attention.head_count"))
	if numHeads == 0 {
		for i := range numLayers {
			if i < len(headCount) && i < len(headCountKV) && headCount[i] > 0 && headCountKV[i] > 0 {
				numHeads = int(headCount[i])
				break
			}
		}
	}
	numKVHeads := int(c.Uint("attention.head_count_kv"))
	if numKVHeads == 0 {
		for i := range numLayers {
			if i < len(headCountKV) && i < len(ffnLength) && headCountKV[i] > 0 && ffnLength[i] == 0 {
				numKVHeads = int(headCountKV[i])
				break
			}
		}
		if numKVHeads == 0 {
			numKVHeads = numHeads
		}
	}

	headDim := int(c.Uint("attention.head_dim"))
	if headDim == 0 {
		if keyLength := int(c.Uint("attention.key_length")); keyLength > 0 {
			headDim = keyLength
		} else if numHeads > 0 {
			headDim = int(c.Uint("embedding_length")) / numHeads
		}
	}
	if headDim <= 0 {
		return nil, fmt.Errorf("nemotronh: invalid attention head dimension")
	}
	if numHeads <= 0 {
		// Attention layers derive per-layer head counts from projection weights.
		// Keep a non-zero default to avoid invalid option math.
		numHeads = 1
	}

	numExperts := int(c.Uint("expert_count"))
	numExpertsUsed := int(c.Uint("expert_used_count"))
	if numExperts > 0 {
		if numExpertsUsed <= 0 || numExpertsUsed > numExperts {
			return nil, fmt.Errorf("nemotronh: invalid expert_used_count=%d for expert_count=%d", numExpertsUsed, numExperts)
		}
	}

	opts := &Options{
		hiddenSize:            int(c.Uint("embedding_length")),
		numHeads:              numHeads,
		numKVHeads:            numKVHeads,
		headDim:               headDim,
		eps:                   c.Float("attention.layer_norm_rms_epsilon"),
		ssmDConv:              int(c.Uint("ssm.conv_kernel")),
		ssmDInner:             int(c.Uint("ssm.inner_size")),
		ssmDState:             int(c.Uint("ssm.state_size")),
		ssmNHead:              int(c.Uint("ssm.time_step_rank")),
		ssmNGroup:             int(c.Uint("ssm.group_count")),
		isRecurrent:           isRecurrent,
		nFF:                   nFF,
		attentionScale:        float64(c.Float("attention.scale")),
		numExperts:            numExperts,
		numExpertsUsed:        numExpertsUsed,
		expertWeightsNorm:     c.Bool("expert_weights_norm", false),
		expertWeightsScale:    c.Float("expert_weights_scale", 1.0),
		expertWeightsNormClip: c.Float("expert_weights_norm_clip", 0),
	}

	// Calculate cache dimensions
	convDim := max(0, opts.ssmDConv-1)
	convChannels := opts.ssmDInner + 2*opts.ssmNGroup*opts.ssmDState
	ssmHeadDim := 0
	if opts.ssmNHead > 0 {
		ssmHeadDim = opts.ssmDInner / opts.ssmNHead
	}
	ssmStateSize := opts.ssmDState * ssmHeadDim * opts.ssmNHead

	m := Model{
		Tokenizer: tokenizer.NewBytePairEncoding(
			&tokenizer.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		Layers:  layers,
		Options: opts,
	}

	m.Cache = NewHybridCache(convDim, convChannels, ssmStateSize)
	return &m, nil
}

func init() {
	model.Register("nemotron_h", New)
	model.Register("nemotron_h_moe", New)
}

// Ensure Model implements model.Model
var _ model.Model = (*Model)(nil)

// Dense implements standard feedforward with ReLU-squared activation
type Dense struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (d *Dense) Forward(ctx ml.Context, x ml.Tensor, opts *Options) ml.Tensor {
	// up -> ReLU-squared -> down
	up := d.Up.Forward(ctx, x)
	up = up.RELU(ctx)
	up = up.Mul(ctx, up) // Square
	return d.Down.Forward(ctx, up)
}

// MoESparse implements MoE with shared experts for Nemotron-H-MoE
type MoESparse struct {
	Router *nn.Linear      `gguf:"ffn_gate_inp"`
	Up     *nn.LinearBatch `gguf:"ffn_up_exps"`
	Down   *nn.LinearBatch `gguf:"ffn_down_exps"`
	Bias   ml.Tensor       `gguf:"exp_probs_b.bias,alt:exp_probs_b"`

	LatentIn  *nn.Linear `gguf:"ffn_latent_in"`
	LatentOut *nn.Linear `gguf:"ffn_latent_out"`

	// Shared experts
	SharedUp   *nn.Linear `gguf:"ffn_up_shexp"`
	SharedDown *nn.Linear `gguf:"ffn_down_shexp"`
}

func (m *MoESparse) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenDim := hiddenStates.Dim(0)
	seqLen := hiddenStates.Dim(1)
	batchSize := hiddenStates.Dim(2)
	if batchSize == 0 {
		batchSize = 1
	}
	hiddenStates2D := hiddenStates.Reshape(ctx, hiddenDim, seqLen*batchSize)

	// Router logits with sigmoid gating
	routerLogits := m.Router.Forward(ctx, hiddenStates2D)

	// Weights come from unbiased sigmoid probabilities.
	probs := routerLogits.Sigmoid(ctx)

	// Selection uses optional bias.
	selectionProbs := probs
	if m.Bias != nil {
		selectionProbs = selectionProbs.Add(ctx, m.Bias)
	}

	// Select top-k experts
	selectedExperts := selectionProbs.TopK(ctx, opts.numExpertsUsed)
	routingWeights := probs.Reshape(ctx, 1, opts.numExperts, hiddenStates2D.Dim(1)).Rows(ctx, selectedExperts)

	// Normalize routing weights
	if opts.expertWeightsNorm {
		routingWeights = routingWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates2D.Dim(1))
		weightsSum := routingWeights.SumRows(ctx)
		weightsSum = weightsSum.Clamp(ctx, 6.103515625e-5, float32(math.MaxFloat32))
		routingWeights = routingWeights.Div(ctx, weightsSum)
		routingWeights = routingWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates2D.Dim(1))
	}

	// Scale routing weights
	if opts.expertWeightsScale != 1.0 {
		routingWeights = routingWeights.Scale(ctx, float64(opts.expertWeightsScale))
	}

	routedInput := hiddenStates2D
	if m.LatentIn != nil {
		routedInput = m.LatentIn.Forward(ctx, routedInput)
	}
	hiddenStates3D := routedInput.Reshape(ctx, routedInput.Dim(0), 1, routedInput.Dim(1))

	// Expert computation with ReLU-squared activation
	upOut := m.Up.Forward(ctx, hiddenStates3D, selectedExperts)
	upOut = upOut.RELU(ctx)
	upOut = upOut.Mul(ctx, upOut) // Square
	experts := m.Down.Forward(ctx, upOut, selectedExperts)
	experts = experts.Mul(ctx, routingWeights)

	// Sum over experts
	moeOut := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		moeOut = moeOut.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}
	if m.LatentOut != nil {
		moeOut = m.LatentOut.Forward(ctx, moeOut)
	}

	// Add shared experts if present
	if m.SharedUp != nil {
		sharedUp := m.SharedUp.Forward(ctx, hiddenStates2D)
		sharedUp = sharedUp.RELU(ctx)
		sharedUp = sharedUp.Mul(ctx, sharedUp) // Square
		sharedOut := m.SharedDown.Forward(ctx, sharedUp)
		moeOut = moeOut.Add(ctx, sharedOut)
	}

	return moeOut
}
