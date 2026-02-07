package qwen3next

import (
	"cmp"
	"fmt"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

// Options contains model configuration
type Options struct {
	hiddenSize  int
	numHeads    int
	numKVHeads  int
	keyLength   int
	valueLength int
	ropeDim     int

	eps                   float32
	ropeBase              float32
	ropeScale             float32
	ropeType              string
	originalContextLength int
	attentionScale        float64

	// MoE config
	numExperts     int
	numExpertsUsed int
	normTopKProb   bool

	// Linear attention (Gated Delta Net) config
	ssmDInner      int // d_inner = head_v_dim * num_v_heads
	ssmDState      int // head_k_dim
	ssmNGroup      int // num_k_heads
	ssmDtRank      int // num_v_heads
	convKernelSize int // SSM conv kernel size

	// Per-layer type from GGUF metadata
	isRecurrent []bool

	// Pre-computed masks for chunked attention (created once per forward pass)
	masks *Masks
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
	ropeDim := cmp.Or(o.ropeDim, o.headDim())
	return nn.RoPE(ctx, states, positions, ropeDim, o.ropeBase, 1./o.ropeScale, opts...)
}

// Operator is the interface for attention-like operators
type Operator interface {
	Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error)
}

// MLP is the interface for feedforward networks
type MLP interface {
	Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor
}

// sparse implements MoE with shared experts
type sparse struct {
	Router *nn.Linear      `gguf:"ffn_gate_inp"`
	Gate   *nn.LinearBatch `gguf:"ffn_gate_exps"`
	Up     *nn.LinearBatch `gguf:"ffn_up_exps"`
	Down   *nn.LinearBatch `gguf:"ffn_down_exps"`

	// Shared experts
	SharedGateInp *nn.Linear `gguf:"ffn_gate_inp_shexp"`
	SharedGate    *nn.Linear `gguf:"ffn_gate_shexp"`
	SharedUp      *nn.Linear `gguf:"ffn_up_shexp"`
	SharedDown    *nn.Linear `gguf:"ffn_down_shexp"`
}

func (mlp *sparse) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
	if batchSize == 0 {
		batchSize = 1
	}
	hiddenStates2D := hiddenStates.Reshape(ctx, hiddenDim, sequenceLength*batchSize)

	// Router logits
	routerLogits := mlp.Router.Forward(ctx, hiddenStates2D)

	// Softmax routing weights
	routingWeights := routerLogits.Softmax(ctx)
	selectedExperts := routingWeights.TopK(ctx, opts.numExpertsUsed)
	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExperts, hiddenStates2D.Dim(1)).Rows(ctx, selectedExperts)
	if opts.normTopKProb {
		routingWeights = routingWeights.Reshape(ctx, opts.numExpertsUsed, hiddenStates2D.Dim(1))
		routingWeights = routingWeights.Div(ctx, routingWeights.SumRows(ctx))
		routingWeights = routingWeights.Reshape(ctx, 1, opts.numExpertsUsed, hiddenStates2D.Dim(1))
	}

	hiddenStates3D := hiddenStates2D.Reshape(ctx, hiddenStates2D.Dim(0), 1, hiddenStates2D.Dim(1))

	// Expert computation with SILU activation
	gateOut := mlp.Gate.Forward(ctx, hiddenStates3D, selectedExperts)
	upOut := mlp.Up.Forward(ctx, hiddenStates3D, selectedExperts)
	experts := gateOut.SILU(ctx, upOut)
	experts = mlp.Down.Forward(ctx, experts, selectedExperts)
	experts = experts.Mul(ctx, routingWeights)

	// Sum over experts
	moeOut := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		moeOut = moeOut.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}

	// Add shared experts if present
	if mlp.SharedUp != nil {
		sharedGate := mlp.SharedGate.Forward(ctx, hiddenStates2D)
		sharedUp := mlp.SharedUp.Forward(ctx, hiddenStates2D)
		sharedOut := sharedGate.SILU(ctx, sharedUp)
		sharedOut = mlp.SharedDown.Forward(ctx, sharedOut)

		// Apply shared expert gating
		if mlp.SharedGateInp != nil {
			sharedGateVal := mlp.SharedGateInp.Forward(ctx, hiddenStates2D)
			sharedGateVal = sharedGateVal.SigmoidOut(ctx)
			// Broadcast gate to match dimensions
			sharedGateVal = sharedGateVal.Repeat(ctx, 0, sharedOut.Dim(0))
			sharedOut = sharedOut.Mul(ctx, sharedGateVal)
		}

		moeOut = moeOut.Add(ctx, sharedOut)
	}

	return moeOut
}

// dense implements standard feedforward
type dense struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *dense) Forward(ctx ml.Context, hiddenStates ml.Tensor, _ *Options) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

// Layer represents a single transformer layer
type Layer struct {
	AttentionNorm     *nn.RMSNorm `gguf:"attn_norm"`
	AttentionPostNorm *nn.RMSNorm `gguf:"post_attention_norm"` // Post-attention norm before FFN
	Operator          Operator

	FFNNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP     MLP
}

func (l *Layer) Forward(ctx ml.Context, layer int, hiddenStates, positions, outputs ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error) {
	residual := hiddenStates

	// Pre-attention norm
	hiddenStates = l.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)

	// Attention (full or linear)
	var err error
	hiddenStates, err = l.Operator.Forward(ctx, hiddenStates, positions, cache, opts)
	if err != nil {
		return nil, err
	}

	// Output projection for last layer
	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	// First residual connection
	hiddenStates = hiddenStates.Add(ctx, residual)

	// Save for FFN residual
	ffnResidual := hiddenStates

	// Post-attention norm (before FFN)
	hiddenStates = l.AttentionPostNorm.Forward(ctx, hiddenStates, opts.eps)

	// FFN
	hiddenStates = l.MLP.Forward(ctx, hiddenStates, opts)

	// Second residual connection
	return hiddenStates.Add(ctx, ffnResidual), nil
}

// Model is the main Qwen3-Next model
type Model struct {
	model.Base
	tokenizer.Tokenizer

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	Layers []Layer `gguf:"blk"`

	*Options
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	cache := m.Cache.(*HybridCache)

	// Create masks once per forward pass
	m.Options.masks = createMasks(ctx)

	for i, layer := range m.Layers {
		cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		var err error
		hiddenStates, err = layer.Forward(ctx, i, hiddenStates, positions, outputs, cache, m.Options)
		if err != nil {
			return nil, err
		}
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

var _ model.Model = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	numLayers := int(c.Uint("block_count"))
	layers := make([]Layer, numLayers)

	// Get per-layer head counts (for detecting layer type)
	type headCounts interface {
		HeadCount() []uint64
		HeadCountKV() []uint64
	}

	var isRecurrent []bool
	var headCountKV []uint64
	if hc, ok := c.(headCounts); ok {
		headCountKV = hc.HeadCountKV()
	}

	isRecurrent = make([]bool, numLayers)
	hasZero := false
	hasFull := false
	for i := range numLayers {
		// If KV head count is 0, it's a recurrent layer
		if i < len(headCountKV) && headCountKV[i] == 0 {
			isRecurrent[i] = true
			hasZero = true
		} else if i < len(headCountKV) && headCountKV[i] > 0 {
			hasFull = true
		}
	}
	if !hasZero || !hasFull {
		return nil, fmt.Errorf("qwen3next: invalid attention.head_count_kv array; expected mix of zero and non-zero values")
	}

	// Determine if MoE
	isMoE := c.Uint("expert_count") > 0

	for i := range layers {
		if isRecurrent[i] {
			layers[i].Operator = &GatedDeltaNet{Layer: i}
		} else {
			layers[i].Operator = &FullAttention{}
		}

		if isMoE {
			layers[i].MLP = &sparse{}
		} else {
			layers[i].MLP = &dense{}
		}
	}

	opts := &Options{
		hiddenSize: int(c.Uint("embedding_length")),
		numHeads:   int(c.Uint("attention.head_count")),
		numKVHeads: func() int {
			for _, v := range headCountKV {
				if v > 0 {
					return int(v)
				}
			}
			return 0
		}(),
		keyLength:             int(c.Uint("attention.key_length")),
		valueLength:           int(c.Uint("attention.value_length")),
		ropeDim:               int(c.Uint("rope.dimension_count")),
		eps:                   c.Float("attention.layer_norm_rms_epsilon"),
		ropeType:              c.String("rope.scaling.type"),
		ropeBase:              c.Float("rope.freq_base"),
		ropeScale:             c.Float("rope.scaling.factor", 1),
		originalContextLength: int(c.Uint("rope.scaling.original_context_length")),
		attentionScale:        float64(c.Float("attention.scale")),
		numExperts:            int(c.Uint("expert_count")),
		numExpertsUsed:        int(c.Uint("expert_used_count")),
		normTopKProb:          c.Bool("norm_top_k_prob", true),
		ssmDInner:             int(c.Uint("ssm.inner_size")),
		ssmDState:             int(c.Uint("ssm.state_size")),
		ssmNGroup:             int(c.Uint("ssm.group_count")),
		ssmDtRank:             int(c.Uint("ssm.time_step_rank")),
		convKernelSize:        int(c.Uint("ssm.conv_kernel")),
		isRecurrent:           isRecurrent,
	}
	if opts.numKVHeads == 0 {
		return nil, fmt.Errorf("qwen3next: attention.head_count_kv array must include at least one non-zero value")
	}

	// Calculate cache dimensions
	convDim := max(0, opts.convKernelSize-1)
	convChannels := opts.ssmDInner + 2*opts.ssmNGroup*opts.ssmDState
	headVDim := 0
	numVHeads := opts.ssmDtRank
	if numVHeads > 0 {
		headVDim = opts.ssmDInner / numVHeads
	}
	deltaStateSize := headVDim * headVDim * numVHeads

	// Validate dimension assumption: headKDim == headVDim is required for state computations
	headKDim := opts.ssmDState
	if headKDim != headVDim && headKDim > 0 && headVDim > 0 {
		return nil, fmt.Errorf("qwen3next: headKDim (%d) != headVDim (%d) not supported; state computations require equal dimensions", headKDim, headVDim)
	}

	m := Model{
		Tokenizer: tokenizer.NewBytePairEncoding(
			&tokenizer.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				// Qwen3 tokenizers typically set add_bos_token=false and bos_token=null.
				// Default to false when the GGUF key is missing to avoid injecting a spurious BOS.
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

	m.Cache = NewHybridCache(m.Shift, convDim, convChannels, deltaStateSize)
	return &m, nil
}

func init() {
	model.Register("qwen3next", New)
}
