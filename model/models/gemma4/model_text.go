package gemma4

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model/input"
)

const (
	cacheTypeSWA = iota
	cacheTypeCausal
)

type TextOptions struct {
	hiddenSize              int
	numHeads, numKVHeads    int
	numGlobalKVHeads        int
	headDim, globalHeadDim  int
	hiddenLayers            int
	hiddenSizePerLayerInput int

	eps               float32
	ropeBase          float32
	ropeLocalBase     float32
	partialRotaryDims int // RoPE dims for full-attention (global) layers

	slidingWindowPattern []bool
	// kvDonorMap maps shared layer index -> donor layer index.
	// Donor is the last non-shared layer of the same type (sliding/full).
	kvDonorMap map[int]int

	finalLogitSoftcap float32

	numExperts     int
	numExpertsUsed int
}

func (o *TextOptions) isLocal(layer int) bool {
	if layer < len(o.slidingWindowPattern) {
		return o.slidingWindowPattern[layer]
	}
	return false
}

func (o *TextOptions) ropeForLayer(layer int) (base float32, dims int) {
	if o.isLocal(layer) {
		return o.ropeLocalBase, o.headDim
	}
	return o.ropeBase, o.partialRotaryDims
}

func (o *TextOptions) kvHeadsForLayer(layer int) int {
	if o.isLocal(layer) {
		return o.numKVHeads
	}
	if o.numGlobalKVHeads > 0 {
		return o.numGlobalKVHeads
	}
	return o.numKVHeads
}

func (o *TextOptions) headDimForLayer(layer int) int {
	if o.isLocal(layer) {
		return o.headDim
	}
	return o.globalHeadDim
}

type TextModel struct {
	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	*PerLayerProjector
	Layers     []TextLayer `gguf:"blk"`
	OutputNorm *nn.RMSNorm `gguf:"output_norm"`
	Output     *nn.Linear  `gguf:"output,alt:token_embd"`
	TextOptions
}

func newTextModel(c fs.Config) *TextModel {
	numLayers := int(c.Uint("block_count"))

	// Head dimensions: key_length is global head dim, key_length_swa is local (SWA) head dim.
	globalHeadDim := int(c.Uint("attention.key_length", 512))
	headDim := int(c.Uint("attention.key_length_swa", 256))

	// RoPE dimensions for global (full attention) layers with proportional RoPE.
	// The freq_factors tensor handles partial rotation (1.0 for rotated pairs,
	// 1e30 for non-rotated), so ropeDims equals the full global head dim.
	partialRotaryDims := int(c.Uint("rope.dimension_count", 0))
	if partialRotaryDims == 0 {
		partialFactor := c.Float("rope.partial_rotary_factor", 1.0)
		partialRotaryDims = int(float32(globalHeadDim) * partialFactor)
	}

	ropeBase := c.Float("rope.freq_base", 1000000.0)
	ropeLocalBase := c.Float("rope.freq_base_swa", 0)
	if ropeLocalBase == 0 {
		ropeLocalBase = c.Float("rope.local.freq_base", 10000.0)
	}

	numGlobalKVHeads := int(c.Uint("attention.global_head_count_kv", 0))
	slidingPattern := c.Bools("attention.sliding_window_pattern")

	// KV heads: try per-layer array first (MoE models), then fall back to scalar
	numKVHeads := 0
	kvHeadsArray := c.Ints("attention.head_count_kv")
	if len(kvHeadsArray) > 0 {
		numKVHeads = int(kvHeadsArray[0])
		if numGlobalKVHeads == 0 && len(slidingPattern) > 0 {
			for i, isLocal := range slidingPattern {
				if !isLocal && i < len(kvHeadsArray) {
					numGlobalKVHeads = int(kvHeadsArray[i])
					break
				}
			}
		}
	}
	if numKVHeads == 0 {
		numKVHeads = int(c.Uint("attention.head_count_kv", 0))
	}

	// Compute KV sharing donor map (same logic as MLX)
	sharedLayers := int(c.Uint("attention.shared_kv_layers", 0))
	kvDonorMap := make(map[int]int)
	if sharedLayers > 0 && len(slidingPattern) > 0 {
		firstShared := numLayers - sharedLayers
		for i := firstShared; i < numLayers; i++ {
			isLocal := slidingPattern[i]
			// Find last non-shared layer of same type
			for j := firstShared - 1; j >= 0; j-- {
				if slidingPattern[j] == isLocal {
					kvDonorMap[i] = j
					break
				}
			}
		}
	}

	return &TextModel{
		Layers: make([]TextLayer, numLayers),
		TextOptions: TextOptions{
			hiddenSize:              int(c.Uint("embedding_length")),
			numHeads:                int(c.Uint("attention.head_count")),
			numKVHeads:              numKVHeads,
			numGlobalKVHeads:        numGlobalKVHeads,
			headDim:                 headDim,
			globalHeadDim:           globalHeadDim,
			hiddenLayers:            numLayers,
			hiddenSizePerLayerInput: int(c.Uint("embedding_length_per_layer_input", 0)),
			eps:                     c.Float("attention.layer_norm_rms_epsilon", 1e-06),
			ropeBase:                ropeBase,
			ropeLocalBase:           ropeLocalBase,
			partialRotaryDims:       partialRotaryDims,
			slidingWindowPattern:    slidingPattern,
			kvDonorMap:              kvDonorMap,
			finalLogitSoftcap:       c.Float("final_logit_softcapping", 0.0),
			numExperts:              int(c.Uint("expert_count", 0)),
			numExpertsUsed:          int(c.Uint("expert_used_count", 0)),
		},
	}
}

func (m *TextModel) Forward(ctx ml.Context, batch input.Batch, cache kvcache.Cache) ml.Tensor {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	hiddenState = hiddenState.Scale(ctx, math.Sqrt(float64(m.hiddenSize)))

	// Inject vision embeddings into the hidden state
	var except []int
	for _, image := range batch.Multimodal {
		visionOutputs := image.Multimodal[0].Tensor
		ctx.Forward(visionOutputs.Copy(ctx, hiddenState.View(ctx, image.Index*hiddenState.Stride(1), visionOutputs.Dim(0)*visionOutputs.Dim(1))))

		for i := range visionOutputs.Dim(1) {
			except = append(except, image.Index+i)
		}
	}

	// PLE
	var perLayerInputs ml.Tensor
	if m.PerLayerProjector != nil {
		perLayerInputs = m.PerLayerProjector.Forward(ctx, batch, hiddenState, &m.TextOptions)
	}

	for i := range len(m.Layers) {
		layer := m.Layers[i]
		if cache != nil {
			cache.SetLayer(i)
			cacheType := cacheTypeSWA
			if !m.isLocal(i) {
				cacheType = cacheTypeCausal
			}
			wc := cache.(*kvcache.WrapperCache)
			wc.SetLayerType(cacheType)

			if causal, ok := wc.UnderlyingCache().(*kvcache.Causal); ok {
				causal.SetCausal(ctx, kvcache.CausalOptions{Except: except})
			}
		}

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = batch.Outputs
		}

		var perLayerInput ml.Tensor
		if perLayerInputs != nil {
			perLayerInput = perLayerInputs.View(ctx, i*perLayerInputs.Stride(1), perLayerInputs.Dim(0), perLayerInputs.Stride(2), perLayerInputs.Dim(2))
		}

		// KV sharing: layers >= firstShared reuse K/V from donor layers
		isShared := false
		if donorLayer, ok := m.kvDonorMap[i]; ok {
			// Set cache layer to donor so Get() reads donor's K/V
			cache.SetLayer(donorLayer)
			isShared = true
		}
		hiddenState = layer.Forward(ctx, i, hiddenState, positions, perLayerInput, lastLayerOutputs, cache, isShared, &m.TextOptions)
	}

	return m.OutputNorm.Forward(ctx, hiddenState, m.eps)
}

// PerLayerProjector implements PLE.
type PerLayerProjector struct {
	TokenEmbedding *nn.Embedding `gguf:"per_layer_token_embd"`
	Projector      *nn.Linear    `gguf:"per_layer_model_proj"`
	Norm           *nn.RMSNorm   `gguf:"per_layer_proj_norm"`
}

func (p *PerLayerProjector) Forward(ctx ml.Context, batch input.Batch, inputs ml.Tensor, opts *TextOptions) ml.Tensor {
	inputsPerLayer := p.TokenEmbedding.Forward(ctx, batch.Inputs)
	inputsPerLayer = inputsPerLayer.Scale(ctx, math.Sqrt(float64(opts.hiddenSizePerLayerInput)))
	// Reshape to [pleDim, numLayers, numTokens] — matching projection shape
	inputsPerLayer = inputsPerLayer.Reshape(ctx, opts.hiddenSizePerLayerInput, opts.hiddenLayers, inputs.Dim(1))

	perLayerProjection := p.Projector.Forward(ctx, inputs)
	perLayerProjection = perLayerProjection.Scale(ctx, 1.0/math.Sqrt(float64(opts.hiddenSize)))
	perLayerProjection = perLayerProjection.Reshape(ctx, opts.hiddenSizePerLayerInput, opts.hiddenLayers, inputs.Dim(1))
	perLayerProjection = p.Norm.Forward(ctx, perLayerProjection, opts.eps)

	if inputsPerLayer != nil {
		perLayerProjection = perLayerProjection.Add(ctx, inputsPerLayer)
		perLayerProjection = perLayerProjection.Scale(ctx, 1/math.Sqrt(2))
	}

	return perLayerProjection
}

type TextSelfAttention struct {
	Query       *nn.Linear  `gguf:"attn_q"`
	QueryNorm   *nn.RMSNorm `gguf:"attn_q_norm"`
	Key         *nn.Linear  `gguf:"attn_k"`
	KeyNorm     *nn.RMSNorm `gguf:"attn_k_norm"`
	Value       *nn.Linear  `gguf:"attn_v"`
	Output      *nn.Linear  `gguf:"attn_output"`
	RopeFactors ml.Tensor   `gguf:"rope_freqs.weight"` // proportional RoPE freq_factors
}

func (sa *TextSelfAttention) Forward(ctx ml.Context, layer int, hiddenState, positions ml.Tensor, cache kvcache.Cache, sharedKV bool, opts *TextOptions) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	hd := opts.headDimForLayer(layer)
	kvHeads := opts.kvHeadsForLayer(layer)
	ropeBase, ropeDims := opts.ropeForLayer(layer)

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, hd, opts.numHeads, batchSize)
	q = sa.QueryNorm.Forward(ctx, q, opts.eps)

	var k, v ml.Tensor
	if !sharedKV {
		k = sa.Key.Forward(ctx, hiddenState)
		k = k.Reshape(ctx, hd, kvHeads, batchSize)

		if sa.Value != nil {
			v = sa.Value.Forward(ctx, hiddenState)
			v = v.Reshape(ctx, hd, kvHeads, batchSize)
		} else {
			// K=V: use raw K projection (before K norm) as V
			v = k
		}

		k = sa.KeyNorm.Forward(ctx, k, opts.eps)
		v = v.RMSNorm(ctx, nil, opts.eps) // V norm: unweighted RMSNorm
	}

	// RoPE with proportional freq_factors on global layers
	ropeOpts := []func(*rope.Options){rope.WithTypeNeoX()}
	if sa.RopeFactors != nil && !opts.isLocal(layer) {
		ropeOpts = append(ropeOpts, rope.WithFactors(sa.RopeFactors))
	}
	q = nn.RoPE(ctx, q, positions, ropeDims, ropeBase, 1.0, ropeOpts...)
	if k != nil {
		k = nn.RoPE(ctx, k, positions, ropeDims, ropeBase, 1.0, ropeOpts...)
	}

	attention := nn.Attention(ctx, q, k, v, 1.0, cache)

	attention = attention.Reshape(ctx, hd*opts.numHeads, batchSize)
	return sa.Output.Forward(ctx, attention)
}

type TextMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).GELU(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

// TextRouter implements the Gemma 4 MoE router.
type TextRouter struct {
	Proj           *nn.Linear `gguf:"ffn_gate_inp"`
	Scale          ml.Tensor  `gguf:"ffn_gate_inp.scale"`
	PerExpertScale ml.Tensor  `gguf:"ffn_gate_inp.per_expert_scale"`
}

func (r *TextRouter) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextOptions) (routingWeights, selectedExperts ml.Tensor) {
	// RMSNorm without learned weight
	x := hiddenState.RMSNorm(ctx, nil, opts.eps)
	// Scale by 1/sqrt(hidden_size)
	x = x.Scale(ctx, 1.0/math.Sqrt(float64(opts.hiddenSize)))
	// Multiply by learned scale parameter
	x = x.Mul(ctx, r.Scale)
	// Project to expert logits
	expertScores := r.Proj.Forward(ctx, x)
	// Softmax over experts
	routingWeights = expertScores.Softmax(ctx)
	// TopK expert selection
	selectedExperts = routingWeights.TopK(ctx, opts.numExpertsUsed)
	return routingWeights, selectedExperts
}

// TextMoEBlock implements the Gemma 4 sparse MoE.
type TextMoEBlock struct {
	Gate *nn.LinearBatch `gguf:"ffn_gate_exps"`
	Up   *nn.LinearBatch `gguf:"ffn_up_exps"`
	Down *nn.LinearBatch `gguf:"ffn_down_exps"`
}

func (moe *TextMoEBlock) Forward(ctx ml.Context, hiddenState, routingWeights, selectedExperts ml.Tensor, perExpertScale ml.Tensor, opts *TextOptions) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	rw3d := routingWeights.Reshape(ctx, 1, opts.numExperts, batchSize)

	// Gather per-expert scales for the selected experts (before we consume routingWeights).
	// Multiply scale [numExperts] into the full softmax [numExperts, batchSize],
	// then use Rows to select — gives us [numExpertsUsed, batchSize] of (softmax*scale).
	// After we independently renorm the unscaled weights, we divide to recover just the scales.
	// But that's fragile. Instead: just Mul scale into softmax, Rows both, renorm unscaled,
	// then Mul the ratio.
	//
	// Simpler: Mul per_expert_scale onto the softmax ONCE to create a scaled copy.
	// Rows both. Renorm the unscaled. Divide scaled/unscaled_pre_renorm to get scale[k].
	// Mul scale[k] * renormed.
	//
	// Actually simplest: just select, renorm, then Mul the scale factors.
	// To get scale factors for selected experts, Rows on a [1, numExperts, batchSize]
	// tensor where every column is the same scale. Build it via Mul: softmax * scale
	// gives us the scaled version; softmax gives unscaled. Their ratio = scale[k].
	// After Rows, ratio = (softmax[k]*scale[k]) / softmax[k] = scale[k].

	// Select routing weights for chosen experts and renormalize
	routingWeights = rw3d.Rows(ctx, selectedExperts)
	routingWeights = routingWeights.Reshape(ctx, opts.numExpertsUsed, batchSize)
	routingWeights = routingWeights.Div(ctx, routingWeights.SumRows(ctx))

	// Apply per-expert scale after renormalization
	if perExpertScale != nil {
		// Build [numExperts, batchSize] with scale broadcast, select with Rows
		scaledSoftmax := rw3d.Reshape(ctx, opts.numExperts, batchSize).Mul(ctx, perExpertScale)
		scaledSoftmax = scaledSoftmax.Reshape(ctx, 1, opts.numExperts, batchSize).Rows(ctx, selectedExperts)
		scaledSoftmax = scaledSoftmax.Reshape(ctx, opts.numExpertsUsed, batchSize)

		// Recover unscaled selected weights (before renorm) for the ratio
		unscaled := rw3d.Rows(ctx, selectedExperts)
		unscaled = unscaled.Reshape(ctx, opts.numExpertsUsed, batchSize)

		// scale[k] = scaledSoftmax[k] / unscaled[k]
		expertScales := scaledSoftmax.Div(ctx, unscaled)
		routingWeights = routingWeights.Mul(ctx, expertScales)
	}

	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExpertsUsed, batchSize)

	hiddenState = hiddenState.Reshape(ctx, hiddenState.Dim(0), 1, hiddenState.Dim(1))

	// Expert computation using LinearBatch (MulmatID selecting experts by index)
	gateOut := moe.Gate.Forward(ctx, hiddenState, selectedExperts)
	upOut := moe.Up.Forward(ctx, hiddenState, selectedExperts)
	hiddenState = gateOut.GELU(ctx, upOut)
	experts := moe.Down.Forward(ctx, hiddenState, selectedExperts)

	// Apply routing weights
	experts = experts.Mul(ctx, routingWeights)

	// Sum across experts
	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}

	return nextStates
}

type TextLayer struct {
	AttentionNorm     *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention     *TextSelfAttention
	PostAttentionNorm *nn.RMSNorm `gguf:"post_attention_norm,alt:attn_post_norm"`
	MLPNorm           *nn.RMSNorm `gguf:"ffn_norm,alt:ffn_pre_norm"`
	MLP               *TextMLP
	PostMLPNorm       *nn.RMSNorm `gguf:"post_ffw_norm,alt:ffn_post_norm"`

	// MoE (present only for models with enable_moe_block=true)
	Router       *TextRouter
	MoE          *TextMoEBlock
	MoENorm      *nn.RMSNorm `gguf:"pre_ffw_norm_2,alt:ffn_pre_norm_2"`
	PostMoENorm  *nn.RMSNorm `gguf:"post_ffw_norm_2,alt:ffn_post_norm_2"`
	PostMLPNorm1 *nn.RMSNorm `gguf:"post_ffw_norm_1,alt:ffn_post_norm_1"` // used instead of PostMLPNorm when MoE is present

	PerLayerInputGate  *nn.Linear  `gguf:"inp_gate"`
	PerLayerProjection *nn.Linear  `gguf:"proj"`
	PostPerLayerNorm   *nn.RMSNorm `gguf:"post_norm"`
	LayerScalar        ml.Tensor   `gguf:"layer_scalar,alt:layer_output_scale.weight"`
}

func (l *TextLayer) Forward(ctx ml.Context, layer int, hiddenState, positions, perLayerInput, outputs ml.Tensor, cache kvcache.Cache, sharedKV bool, opts *TextOptions) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, layer, hiddenState, positions, cache, sharedKV, opts)
	hiddenState = l.PostAttentionNorm.Forward(ctx, hiddenState, opts.eps)

	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
		if perLayerInput != nil {
			perLayerInput = perLayerInput.Rows(ctx, outputs)
		}
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	// MLP (+ optional MoE in parallel)
	if l.Router != nil && l.MoE != nil && l.MoE.Gate != nil && l.MoE.Gate.Weight != nil {
		// MoE layers: run MLP and MoE in parallel, sum results
		mlpState := l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
		mlpState = l.MLP.Forward(ctx, mlpState)
		mlpState = l.PostMLPNorm1.Forward(ctx, mlpState, opts.eps)

		routingWeights, selectedExperts := l.Router.Forward(ctx, hiddenState, opts)
		moeState := l.MoENorm.Forward(ctx, hiddenState, opts.eps)
		moeState = l.MoE.Forward(ctx, moeState, routingWeights, selectedExperts, l.Router.PerExpertScale, opts)
		moeState = l.PostMoENorm.Forward(ctx, moeState, opts.eps)

		// Combine MLP + MoE, apply outer post-FFN norm, then add residual
		combined := mlpState.Add(ctx, moeState)
		combined = l.PostMLPNorm.Forward(ctx, combined, opts.eps)
		hiddenState = combined.Add(ctx, residual)
	} else {
		// Dense layers: MLP only
		hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
		hiddenState = l.MLP.Forward(ctx, hiddenState)
		hiddenState = l.PostMLPNorm.Forward(ctx, hiddenState, opts.eps)
		hiddenState = hiddenState.Add(ctx, residual)
	}

	// PLE injection (after MLP residual)
	if perLayerInput != nil && l.PerLayerInputGate != nil {
		pleState := l.PerLayerInputGate.Forward(ctx, hiddenState)
		pleState = pleState.GELU(ctx, perLayerInput)
		pleState = l.PerLayerProjection.Forward(ctx, pleState)
		pleState = l.PostPerLayerNorm.Forward(ctx, pleState, opts.eps)
		hiddenState = hiddenState.Add(ctx, pleState)
	}

	// Layer scalar applied at end of layer (full-attention layers only)
	if l.LayerScalar != nil {
		hiddenState = hiddenState.Mul(ctx, l.LayerScalar)
	}

	return hiddenState
}
