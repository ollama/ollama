//go:build mlx

// Package glm4_moe_lite provides the GLM4-MoE-Lite implementation for MLX.
// This model uses Multi-head Latent Attention (MLA) and Mixture of Experts (MoE).
package glm4_moe_lite

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// RopeScaling holds RoPE scaling configuration
type RopeScaling struct {
	Factor       float32 `json:"factor"`
	MscaleAllDim float32 `json:"mscale_all_dim"`
}

// Config holds GLM4-MoE-Lite model configuration
type Config struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	MoEIntermediateSize   int32   `json:"moe_intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	AttentionBias         bool    `json:"attention_bias"`

	// MLA (Multi-head Latent Attention) parameters
	QLoraRank     int32 `json:"q_lora_rank"`
	KVLoraRank    int32 `json:"kv_lora_rank"`
	QKRopeHeadDim int32 `json:"qk_rope_head_dim"`
	QKNopeHeadDim int32 `json:"qk_nope_head_dim"`
	VHeadDim      int32 `json:"v_head_dim"`

	// MoE parameters
	NRoutedExperts      int32   `json:"n_routed_experts"`
	NSharedExperts      int32   `json:"n_shared_experts"`
	NumExpertsPerTok    int32   `json:"num_experts_per_tok"`
	RoutedScalingFactor float32 `json:"routed_scaling_factor"`
	NormTopKProb        bool    `json:"norm_topk_prob"`
	FirstKDenseReplace  int32   `json:"first_k_dense_replace"`
	NGroup              int32   `json:"n_group"`
	TopKGroup           int32   `json:"topk_group"`

	// RoPE scaling
	RopeScaling *RopeScaling `json:"rope_scaling"`

	// Quantization parameters (set during load based on model quantization)
	QuantGroupSize int    `json:"-"` // Group size for quantization (default 64)
	QuantBits      int    `json:"-"` // Bits per weight (4 or 8)
	QuantMode      string `json:"-"` // Quantization mode ("affine", etc.)

	// Computed fields
	QHeadDim int32   `json:"-"` // qk_nope_head_dim + qk_rope_head_dim
	Scale    float32 `json:"-"` // 1/sqrt(QHeadDim) with mscale adjustment
}

// MLAAttention implements Multi-head Latent Attention with absorption.
// This uses absorbed MLA which operates in latent space for reduced KV cache.
type MLAAttention struct {
	// Low-rank query projections
	QAProj      nn.LinearLayer `weight:"self_attn.q_a_proj"`
	QALayerNorm *nn.RMSNorm    `weight:"self_attn.q_a_layernorm"`
	QBProj      nn.LinearLayer `weight:"self_attn.q_b_proj"`

	// Low-rank KV projections (with shared rope component)
	KVAProjWithMQA nn.LinearLayer `weight:"self_attn.kv_a_proj_with_mqa"`
	KVALayerNorm   *nn.RMSNorm    `weight:"self_attn.kv_a_layernorm"`

	// Absorbed MLA projections (derived from kv_b_proj)
	// EmbedQ: projects q_nope to latent space [num_heads, kv_lora_rank, qk_nope_head_dim]
	// UnembedOut: projects attention output from latent space [num_heads, v_head_dim, kv_lora_rank]
	EmbedQ     *nn.MultiLinear `weight:"-"`
	UnembedOut *nn.MultiLinear `weight:"-"`

	// Output projection
	OProj nn.LinearLayer `weight:"self_attn.o_proj"`
}

// Forward computes absorbed MLA attention output.
// This operates in latent space for reduced KV cache memory.
func (a *MLAAttention) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	// Query path: q_a_proj -> layernorm -> q_b_proj
	q := a.QAProj.Forward(x)
	q = a.QALayerNorm.Forward(q, cfg.RMSNormEps)
	q = a.QBProj.Forward(q)

	// Reshape Q: [B, L, num_heads * q_head_dim] -> [B, num_heads, L, q_head_dim]
	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.QHeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)

	// Split Q into nope and rope parts
	qNope := mlx.Slice(q, []int32{0, 0, 0, 0}, []int32{B, cfg.NumAttentionHeads, L, cfg.QKNopeHeadDim})
	qPE := mlx.Slice(q, []int32{0, 0, 0, cfg.QKNopeHeadDim}, []int32{B, cfg.NumAttentionHeads, L, cfg.QHeadDim})

	// KV path: get compressed KV and k_pe
	compressedKV := a.KVAProjWithMQA.Forward(x)

	// Split into compressed_kv and k_pe (shared rope component)
	kvCompressed := mlx.Slice(compressedKV, []int32{0, 0, 0}, []int32{B, L, cfg.KVLoraRank})
	kPE := mlx.Slice(compressedKV, []int32{0, 0, cfg.KVLoraRank}, []int32{B, L, cfg.KVLoraRank + cfg.QKRopeHeadDim})

	// k_pe is shared across heads (MQA-style): [B, L, rope_dim] -> [B, 1, L, rope_dim]
	kPE = mlx.Reshape(kPE, B, L, 1, cfg.QKRopeHeadDim)
	kPE = mlx.Transpose(kPE, 0, 2, 1, 3)

	// Apply layernorm to get kv latent representation
	kvLatent := a.KVALayerNorm.Forward(kvCompressed, cfg.RMSNormEps)
	// kvLatent: [B, L, kv_lora_rank] -> [B, 1, L, kv_lora_rank] for broadcasting
	kvLatent = mlx.ExpandDims(kvLatent, 1)

	// Apply RoPE to the rope parts
	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	qPE = mlx.RoPE(qPE, int(cfg.QKRopeHeadDim), true, cfg.RopeTheta, 1.0, offset)
	kPE = mlx.RoPE(kPE, int(cfg.QKRopeHeadDim), true, cfg.RopeTheta, 1.0, offset)

	// ABSORBED MLA: project q_nope to latent space
	// qNope: [B, num_heads, L, qk_nope_head_dim]
	// EmbedQ: [num_heads, kv_lora_rank, qk_nope_head_dim]
	// Result: [B, num_heads, L, kv_lora_rank]
	qLatent := a.EmbedQ.Forward(qNope)

	// Keys = concat(kvLatent, kPE)
	// kvLatent: [B, 1, L, kv_lora_rank]
	// kPE: [B, 1, L, qk_rope_head_dim]
	// keys: [B, 1, L, kv_lora_rank + qk_rope_head_dim]
	keys := mlx.Concatenate([]*mlx.Array{kvLatent, kPE}, 3)

	// Cache the smaller latent representation
	// We cache keys (latent + rope) and use empty values since values are derived from keys
	cachedL := L
	if c != nil {
		// Create placeholder values with 0 dims for cache (we don't actually use cached values)
		placeholderValues := mlx.Zeros([]int32{B, 1, L, 0}, mlx.DtypeFloat32)
		keys, _ = c.Update(keys, placeholderValues, int(L))
		cachedL = int32(keys.Shape()[2])
	}

	// Values are the first kv_lora_rank dims of keys (slice off rope part)
	values := mlx.Slice(keys, []int32{0, 0, 0, 0}, []int32{B, 1, cachedL, cfg.KVLoraRank})

	// Queries = concat(qLatent, qPE)
	// qLatent: [B, num_heads, L, kv_lora_rank]
	// qPE: [B, num_heads, L, qk_rope_head_dim]
	// queries: [B, num_heads, L, kv_lora_rank + qk_rope_head_dim]
	queries := mlx.Concatenate([]*mlx.Array{qLatent, qPE}, 3)

	// Attention in latent space
	// queries: [B, num_heads, L, kv_lora_rank + rope_dim]
	// keys: [B, 1, cachedL, kv_lora_rank + rope_dim]
	// values: [B, 1, cachedL, kv_lora_rank]
	out := mlx.ScaledDotProductAttention(queries, keys, values, cfg.Scale, L > 1)

	// ABSORBED MLA: unembed from latent space
	// out: [B, num_heads, L, kv_lora_rank]
	// UnembedOut: [num_heads, v_head_dim, kv_lora_rank]
	// Result: [B, num_heads, L, v_head_dim]
	out = a.UnembedOut.Forward(out)

	// Reshape back: [B, num_heads, L, v_head_dim] -> [B, L, num_heads * v_head_dim]
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.VHeadDim)

	return a.OProj.Forward(out)
}

// DenseMLP implements the standard SwiGLU MLP for dense layers
type DenseMLP struct {
	GateProj nn.LinearLayer `weight:"mlp.gate_proj"`
	UpProj   nn.LinearLayer `weight:"mlp.up_proj"`
	DownProj nn.LinearLayer `weight:"mlp.down_proj"`
}

// Forward applies the SwiGLU MLP
func (m *DenseMLP) Forward(x *mlx.Array) *mlx.Array {
	gate := mlx.SiLU(m.GateProj.Forward(x))
	up := m.UpProj.Forward(x)
	return m.DownProj.Forward(mlx.Mul(gate, up))
}

// MoEGate implements the expert gating mechanism
type MoEGate struct {
	Gate                 nn.LinearLayer `weight:"mlp.gate"`
	EScoreCorrectionBias *mlx.Array     `weight:"mlp.gate.e_score_correction_bias,optional"`
}

// Forward computes expert selection indices and scores
func (g *MoEGate) Forward(x *mlx.Array, cfg *Config) (*mlx.Array, *mlx.Array) {
	// Compute gate logits through linear layer (handles both quantized and non-quantized)
	gates := g.Gate.Forward(x)

	// Sigmoid scoring
	scores := mlx.Sigmoid(gates)
	origScores := scores

	// Add correction bias if present
	if g.EScoreCorrectionBias != nil {
		scores = mlx.Add(scores, g.EScoreCorrectionBias)
	}

	// Group-wise expert selection (simplified for n_group=1)
	// Select top-k experts
	topK := cfg.NumExpertsPerTok
	negScores := mlx.Neg(scores)
	inds := mlx.Argpartition(negScores, int(topK)-1, -1)

	shape := inds.Shape()
	inds = mlx.Slice(inds, []int32{0, 0, 0}, []int32{shape[0], shape[1], topK})

	// Get scores for selected experts
	scores = mlx.TakeAlongAxis(origScores, inds, -1)

	// Normalize if configured
	if topK > 1 && cfg.NormTopKProb {
		sumScores := mlx.Sum(scores, -1, true)
		scores = mlx.Div(scores, sumScores)
	}

	// Apply routing scaling factor
	scores = mlx.MulScalar(scores, cfg.RoutedScalingFactor)

	return inds, scores
}

// SwitchMLP implements the MoE expert computation using stacked weights
// Note: No weight tags - these are populated manually by stacking expert weights
type SwitchMLP struct {
	// Dequantized weights (used when GatherQMM not available)
	GateWeight *mlx.Array
	UpWeight   *mlx.Array
	DownWeight *mlx.Array

	// Quantized weights (used with GatherQMM for 4/8-bit affine)
	GateWeightQ, GateScales, GateBiases *mlx.Array
	UpWeightQ, UpScales, UpBiases       *mlx.Array
	DownWeightQ, DownScales, DownBiases *mlx.Array

	// Quantization bits per projection (supports mixed precision Q4/Q8)
	GateBits int
	UpBits   int
	DownBits int

	// Quantization group size per projection (detected from tensor shapes)
	GateGroupSize int
	UpGroupSize   int
	DownGroupSize int

	// If true, use GatherQMM with quantized weights
	UseQuantized bool
}

// Forward applies the switched expert MLP
func (s *SwitchMLP) Forward(x *mlx.Array, indices *mlx.Array, cfg *Config) *mlx.Array {
	shape := x.Shape()
	B, L := shape[0], shape[1]
	topK := cfg.NumExpertsPerTok

	// Expand x for expert computation: [B, L, D] -> [B, L, 1, 1, D]
	xExpanded := mlx.ExpandDims(mlx.ExpandDims(x, -2), -2)

	// Flatten for gather_mm: [B*L, 1, 1, D]
	xFlat := mlx.Reshape(xExpanded, B*L, 1, 1, cfg.HiddenSize)

	// Flatten indices: [B, L, topK] -> [B*L, topK]
	idxFlat := mlx.Reshape(indices, B*L, topK)

	// Sort for efficient gather (when we have many tokens)
	doSort := B*L >= 64
	var invOrder *mlx.Array
	n := B * L * topK

	if doSort {
		idxAll := mlx.Flatten(idxFlat)
		order := mlx.Argsort(idxAll, 0)
		invOrder = mlx.Argsort(order, 0)
		// Reorder x based on sorted indices
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), mlx.FloorDivideScalar(order, topK), 0), 1)
		idxFlat = mlx.Reshape(mlx.Take(idxAll, order, 0), n, 1)
	}

	var gate, up, hidden, down *mlx.Array

	if s.UseQuantized {
		// Use GatherQMM for quantized weights (faster, keeps weights quantized)
		// Each projection may have different bits and group sizes (mixed precision: Q4 for gate/up, Q8 for down)
		gate = mlx.GatherQMM(xFlat, s.GateWeightQ, s.GateScales, s.GateBiases,
			nil, idxFlat, true, s.GateGroupSize, s.GateBits, cfg.QuantMode, doSort)
		up = mlx.GatherQMM(xFlat, s.UpWeightQ, s.UpScales, s.UpBiases,
			nil, idxFlat, true, s.UpGroupSize, s.UpBits, cfg.QuantMode, doSort)

		hidden = mlx.Mul(mlx.SiLU(gate), up)

		down = mlx.GatherQMM(hidden, s.DownWeightQ, s.DownScales, s.DownBiases,
			nil, idxFlat, true, s.DownGroupSize, s.DownBits, cfg.QuantMode, doSort)
	} else {
		// Use GatherMM for dequantized/non-quantized weights
		gate = mlx.GatherMM(xFlat, mlx.Transpose(s.GateWeight, 0, 2, 1), nil, idxFlat, doSort)
		up = mlx.GatherMM(xFlat, mlx.Transpose(s.UpWeight, 0, 2, 1), nil, idxFlat, doSort)

		hidden = mlx.Mul(mlx.SiLU(gate), up)

		down = mlx.GatherMM(hidden, mlx.Transpose(s.DownWeight, 0, 2, 1), nil, idxFlat, doSort)
	}

	// Unsort if we sorted
	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

// SharedExperts implements the shared expert MLP
type SharedExperts struct {
	GateProj nn.LinearLayer `weight:"mlp.shared_experts.gate_proj"`
	UpProj   nn.LinearLayer `weight:"mlp.shared_experts.up_proj"`
	DownProj nn.LinearLayer `weight:"mlp.shared_experts.down_proj"`
}

// Forward applies the shared expert MLP
func (s *SharedExperts) Forward(x *mlx.Array) *mlx.Array {
	gate := mlx.SiLU(s.GateProj.Forward(x))
	up := s.UpProj.Forward(x)
	return s.DownProj.Forward(mlx.Mul(gate, up))
}

// MoE implements the full Mixture of Experts layer
type MoE struct {
	Gate          *MoEGate
	SwitchMLP     *SwitchMLP
	SharedExperts *SharedExperts
}

// Forward applies the MoE layer
func (m *MoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	shape := x.Shape()
	B, L := shape[0], shape[1]

	// Get expert indices and scores
	inds, scores := m.Gate.Forward(x, cfg)

	// Apply routed experts
	expertOut := m.SwitchMLP.Forward(x, inds, cfg)

	// Weight by scores: [B, L, topK, D] * [B, L, topK, 1] -> sum over topK
	scoresExpanded := mlx.ExpandDims(scores, -1)
	y := mlx.Sum(mlx.Mul(expertOut, scoresExpanded), 2, false)

	// Add shared experts if present
	if m.SharedExperts != nil {
		y = mlx.Add(y, m.SharedExperts.Forward(x))
	}

	return mlx.Reshape(y, B, L, cfg.HiddenSize)
}

// DenseBlock represents a dense transformer block (for first_k_dense_replace layers)
type DenseBlock struct {
	Attention              *MLAAttention
	MLP                    *DenseMLP
	InputLayerNorm         *nn.RMSNorm `weight:"input_layernorm"`
	PostAttentionLayerNorm *nn.RMSNorm `weight:"post_attention_layernorm"`
}

// Forward applies the dense block
func (b *DenseBlock) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	// Pre-norm attention with residual
	r := b.Attention.Forward(b.InputLayerNorm.Forward(x, cfg.RMSNormEps), c, B, L, cfg)
	h := mlx.Add(x, r)

	// Pre-norm MLP with residual
	r = b.MLP.Forward(b.PostAttentionLayerNorm.Forward(h, cfg.RMSNormEps))
	return mlx.Add(h, r)
}

// MoEBlock represents a MoE transformer block
type MoEBlock struct {
	Attention              *MLAAttention
	MoE                    *MoE
	InputLayerNorm         *nn.RMSNorm `weight:"input_layernorm"`
	PostAttentionLayerNorm *nn.RMSNorm `weight:"post_attention_layernorm"`
}

// Forward applies the MoE block
func (b *MoEBlock) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	// Pre-norm attention with residual
	r := b.Attention.Forward(b.InputLayerNorm.Forward(x, cfg.RMSNormEps), c, B, L, cfg)
	h := mlx.Add(x, r)

	// Pre-norm MoE with residual
	r = b.MoE.Forward(b.PostAttentionLayerNorm.Forward(h, cfg.RMSNormEps), cfg)
	return mlx.Add(h, r)
}

// Block interface for both dense and MoE blocks
type Block interface {
	Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array
}

// Model represents the complete GLM4-MoE-Lite model
type Model struct {
	EmbedTokens *nn.Embedding  `weight:"model.embed_tokens"`
	Layers      []Block        `weight:"-"` // Loaded manually due to different block types
	Norm        *nn.RMSNorm    `weight:"model.norm"`
	LMHead      nn.LinearLayer `weight:"lm_head"`

	tok *tokenizer.Tokenizer
	*Config
}

// computeScale computes the attention scale.
// Uses the full key head dimension (qkNopeHeadDim + qkRopeHeadDim) to match the Ollama runner.
func computeScale(cfg *Config) float32 {
	keyLength := cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	scale := float32(1.0 / math.Sqrt(float64(keyLength)))
	if cfg.RopeScaling != nil && cfg.RopeScaling.MscaleAllDim > 0 && cfg.RopeScaling.Factor > 1 {
		s := 0.1*cfg.RopeScaling.MscaleAllDim*float32(math.Log(float64(cfg.RopeScaling.Factor))) + 1.0
		scale *= s * s
	}
	return scale
}

// supportsGatherQMM returns true if the quantization mode has GatherQMM kernel support.
// Currently only 4-bit and 8-bit affine quantization are supported.
func supportsGatherQMM(mode string, bits int) bool {
	return mode == "affine" && (bits == 4 || bits == 8)
}

// ExpertWeight holds a single expert's weight with optional quantization components.
type ExpertWeight struct {
	Weight    *mlx.Array // Quantized weight (if quantized) or dequantized weight
	Scales    *mlx.Array // Quantization scales (nil if not quantized)
	Biases    *mlx.Array // Quantization biases (nil if not quantized or mode doesn't use biases)
	Bits      int        // Quantization bits (4 or 8), 0 if not quantized
	GroupSize int        // Quantization group size, 0 if not quantized
}

// getQuantParams returns quantization parameters from model metadata.
// Returns groupSize, bits, and mode for the model's quantization type.
func getQuantParams(weights safetensors.WeightSource) (groupSize, bits int, mode string) {
	groupSize, bits, mode = safetensors.QuantizationParams(weights.Quantization())
	// Use metadata group_size if available (overrides default)
	if gs := weights.GroupSize(); gs > 0 {
		groupSize = gs
	}
	return groupSize, bits, mode
}

// loadExpertWeight loads an expert weight.
// If useQuantized is true and the weight is quantized with a supported mode, returns quantized components.
// Otherwise dequantizes and returns only the weight.
func loadExpertWeight(weights safetensors.WeightSource, path string, useQuantized bool, cfg *Config) *ExpertWeight {
	w, _ := weights.GetTensor(path + ".weight")
	if w == nil {
		return nil
	}

	// Check if this is a quantized weight by looking for scales
	scalePath := path + ".weight_scale"
	if weights.HasTensor(scalePath) {
		scales, _ := weights.GetTensor(scalePath)
		var qbiases *mlx.Array
		qbiasPath := path + ".weight_qbias"
		if weights.HasTensor(qbiasPath) {
			qbiases, _ = weights.GetTensor(qbiasPath)
		}

		// Get quantization params from metadata
		groupSize, bits, mode := getQuantParams(weights)

		// Update config with group size (for GatherQMM calls)
		if cfg.QuantGroupSize == 0 {
			cfg.QuantGroupSize = groupSize
		}

		// If GatherQMM is supported and requested, return quantized components
		if useQuantized && supportsGatherQMM(mode, bits) {
			return &ExpertWeight{Weight: w, Scales: scales, Biases: qbiases, Bits: bits, GroupSize: groupSize}
		}

		// Otherwise dequantize
		return &ExpertWeight{Weight: mlx.Dequantize(w, scales, qbiases, groupSize, bits, mode)}
	}

	return &ExpertWeight{Weight: w}
}

// sanitizeMLAWeights transforms kv_b_proj weights into absorbed MLA format.
// Returns embed_q and unembed_out weights for per-head projections.
//
// kv_b_proj.weight shape: [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
// Output:
//   - embed_q: [num_heads, kv_lora_rank, qk_nope_head_dim] - projects q_nope to latent
//   - unembed_out: [num_heads, v_head_dim, kv_lora_rank] - projects latent to output
func sanitizeMLAWeights(weights safetensors.WeightSource, prefix string, cfg *Config) (*mlx.Array, *mlx.Array) {
	path := prefix + ".self_attn.kv_b_proj"
	w, err := weights.GetTensor(path + ".weight")
	if err != nil || w == nil {
		return nil, nil
	}

	// Check if quantized and dequantize
	scalePath := path + ".weight_scale"
	if weights.HasTensor(scalePath) {
		scales, _ := weights.GetTensor(scalePath)
		var qbiases *mlx.Array
		qbiasPath := path + ".weight_qbias"
		if weights.HasTensor(qbiasPath) {
			qbiases, _ = weights.GetTensor(qbiasPath)
		}

		groupSize, bits, mode := getQuantParams(weights)
		w = mlx.Dequantize(w, scales, qbiases, groupSize, bits, mode)
	}

	// w: [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
	// Reshape to [num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank]
	headDim := cfg.QKNopeHeadDim + cfg.VHeadDim
	w = mlx.Reshape(w, cfg.NumAttentionHeads, headDim, cfg.KVLoraRank)

	// Split into wk and wv
	// wk: [num_heads, qk_nope_head_dim, kv_lora_rank]
	// wv: [num_heads, v_head_dim, kv_lora_rank]
	wk := mlx.Slice(w, []int32{0, 0, 0}, []int32{cfg.NumAttentionHeads, cfg.QKNopeHeadDim, cfg.KVLoraRank})
	wv := mlx.Slice(w, []int32{0, cfg.QKNopeHeadDim, 0}, []int32{cfg.NumAttentionHeads, headDim, cfg.KVLoraRank})

	// Transform for absorbed MLA:
	// embed_q: transpose(wk) -> [num_heads, kv_lora_rank, qk_nope_head_dim]
	// This allows: q_nope @ embed_q.T = q_nope @ wk (absorbed key projection)
	embedQ := mlx.Transpose(wk, 0, 2, 1)

	// unembed_out: wv stays [num_heads, v_head_dim, kv_lora_rank]
	// This allows: latent_out @ unembed_out.T = latent_out @ wv.T (absorbed value projection)
	unembedOut := wv

	return embedQ, unembedOut
}

// StackedExpertWeights holds stacked weights for all experts.
type StackedExpertWeights struct {
	Weight    *mlx.Array // Stacked weights [num_experts, out, in] or [num_experts, out, in_packed]
	Scales    *mlx.Array // Stacked scales (nil if not quantized)
	Biases    *mlx.Array // Stacked biases (nil if not quantized)
	Bits      int        // Quantization bits (4 or 8), 0 if not quantized
	GroupSize int        // Quantization group size, 0 if not quantized
}

// collectAndStackExpertWeights loads and stacks expert weights for one projection type.
func collectAndStackExpertWeights(
	weights safetensors.WeightSource,
	prefix string,
	projName string,
	numExperts int32,
	useQuantized bool,
	cfg *Config,
) *StackedExpertWeights {
	var w, s, b []*mlx.Array
	var bits, groupSize int

	for e := int32(0); e < numExperts; e++ {
		path := fmt.Sprintf("%s.mlp.experts.%d.%s", prefix, e, projName)
		ew := loadExpertWeight(weights, path, useQuantized, cfg)
		if ew == nil {
			continue
		}
		w = append(w, ew.Weight)
		if ew.Scales != nil {
			s = append(s, ew.Scales)
		}
		if ew.Biases != nil {
			b = append(b, ew.Biases)
		}
		if e == 0 {
			bits = ew.Bits
			groupSize = ew.GroupSize
		}
	}

	result := &StackedExpertWeights{Bits: bits, GroupSize: groupSize}
	if len(w) > 0 {
		result.Weight = mlx.Stack(w, 0)
		if len(s) > 0 {
			result.Scales = mlx.Stack(s, 0)
		}
		if len(b) > 0 {
			result.Biases = mlx.Stack(b, 0)
		}
	}
	return result
}

// sanitizeExpertWeights stacks individual expert weights into tensors.
// If useQuantized is true and weights support GatherQMM, returns quantized components.
// Otherwise returns dequantized weights with nil scales/biases.
// Bits and GroupSize are detected per-weight to support mixed-precision (Q4 for gate/up, Q8 for down).
func sanitizeExpertWeights(weights safetensors.WeightSource, prefix string, numExperts int32, useQuantized bool, cfg *Config) (gate, up, down *StackedExpertWeights) {
	gate = collectAndStackExpertWeights(weights, prefix, "gate_proj", numExperts, useQuantized, cfg)
	up = collectAndStackExpertWeights(weights, prefix, "up_proj", numExperts, useQuantized, cfg)
	down = collectAndStackExpertWeights(weights, prefix, "down_proj", numExperts, useQuantized, cfg)
	return gate, up, down
}

// LoadFromManifest loads a GLM4-MoE-Lite model from a manifest (Ollama blob storage).
func LoadFromManifest(modelManifest *manifest.ModelManifest) (*Model, error) {
	// Read config from manifest
	configData, err := modelManifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Compute derived fields
	cfg.QHeadDim = cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	cfg.Scale = computeScale(&cfg)

	// Load weights from manifest blobs
	weights, err := manifest.LoadWeightsFromManifest(modelManifest, "")
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	if err := weights.Load(0); err != nil {
		return nil, fmt.Errorf("load weight data: %w", err)
	}

	// Set up quantization parameters (only if model is actually quantized)
	// Note: QuantGroupSize will be detected dynamically from tensor shapes during weight loading
	quantization := weights.Quantization()
	useQuantized := false
	if quantization != "" {
		_, cfg.QuantBits, cfg.QuantMode = safetensors.QuantizationParams(quantization)
		useQuantized = supportsGatherQMM(cfg.QuantMode, cfg.QuantBits)
	}

	// Load tokenizer from manifest with config files for EOS token detection
	tokData, err := modelManifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer config: %w", err)
	}

	// Build tokenizer config with companion files for EOS/BOS token loading
	tokConfig := &tokenizer.TokenizerConfig{
		ConfigJSON: configData, // Already loaded above, contains eos_token_id
	}

	// Try to load generation_config.json if available (preferred source for EOS)
	if genConfigData, err := modelManifest.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = genConfigData
	}

	// Try to load tokenizer_config.json if available
	if tokConfigData, err := modelManifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = tokConfigData
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	m := &Model{
		Layers: make([]Block, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}

	// Load embedding, norm, and lm_head
	if err := safetensors.LoadModule(m, weights, ""); err != nil {
		return nil, err
	}

	// Load layers manually due to different block types
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)

		// Load attention (same for both block types)
		attn := &MLAAttention{}
		if err := safetensors.LoadModule(attn, weights, prefix); err != nil {
			return nil, fmt.Errorf("layer %d attention: %w", i, err)
		}

		// Sanitize MLA weights for absorbed attention
		embedQ, unembedOut := sanitizeMLAWeights(weights, prefix, &cfg)
		attn.EmbedQ = nn.NewMultiLinear(embedQ)
		attn.UnembedOut = nn.NewMultiLinear(unembedOut)

		if i < cfg.FirstKDenseReplace {
			// Dense block
			block := &DenseBlock{Attention: attn}
			if err := safetensors.LoadModule(block, weights, prefix); err != nil {
				return nil, fmt.Errorf("layer %d dense: %w", i, err)
			}
			m.Layers[i] = block
		} else {
			// MoE block
			block := &MoEBlock{Attention: attn}
			if err := safetensors.LoadModule(block, weights, prefix); err != nil {
				return nil, fmt.Errorf("layer %d moe block: %w", i, err)
			}

			// Stack expert weights (pass cfg so group sizes can be detected)
			gate, up, down := sanitizeExpertWeights(weights, prefix, cfg.NRoutedExperts, useQuantized, &cfg)

			switchMLP := &SwitchMLP{UseQuantized: useQuantized}
			if useQuantized {
				switchMLP.GateWeightQ = gate.Weight
				switchMLP.GateScales = gate.Scales
				switchMLP.GateBiases = gate.Biases
				switchMLP.GateBits = gate.Bits
				switchMLP.GateGroupSize = gate.GroupSize
				switchMLP.UpWeightQ = up.Weight
				switchMLP.UpScales = up.Scales
				switchMLP.UpBiases = up.Biases
				switchMLP.UpBits = up.Bits
				switchMLP.UpGroupSize = up.GroupSize
				switchMLP.DownWeightQ = down.Weight
				switchMLP.DownScales = down.Scales
				switchMLP.DownBiases = down.Biases
				switchMLP.DownBits = down.Bits
				switchMLP.DownGroupSize = down.GroupSize
			} else {
				switchMLP.GateWeight = gate.Weight
				switchMLP.UpWeight = up.Weight
				switchMLP.DownWeight = down.Weight
			}

			block.MoE = &MoE{
				Gate:      &MoEGate{},
				SwitchMLP: switchMLP,
			}

			// Load gate weights
			if err := safetensors.LoadModule(block.MoE.Gate, weights, prefix); err != nil {
				return nil, fmt.Errorf("layer %d gate: %w", i, err)
			}

			// Load shared experts if present
			if cfg.NSharedExperts > 0 {
				block.MoE.SharedExperts = &SharedExperts{}
				if err := safetensors.LoadModule(block.MoE.SharedExperts, weights, prefix); err != nil {
					return nil, fmt.Errorf("layer %d shared experts: %w", i, err)
				}
			}

			m.Layers[i] = block
		}
	}

	mlx.Eval(mlx.Collect(m)...)
	weights.ReleaseAll()

	return m, nil
}

// Forward computes the forward pass of the model
func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	B, L := tokens.Shape()[0], tokens.Shape()[1]

	h := m.EmbedTokens.Forward(tokens)

	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil {
			c = caches[i]
		}
		h = layer.Forward(h, c, B, L, m.Config)
	}

	h = m.Norm.Forward(h, m.RMSNormEps)
	return m.LMHead.Forward(h)
}

// Interface methods

// NumLayers returns the number of transformer layers
func (m *Model) NumLayers() int { return len(m.Layers) }

// MaxContextLength returns the maximum context length
func (m *Model) MaxContextLength() int32 { return m.MaxPositionEmbeddings }

// VocabSize returns the vocabulary size
func (m *Model) VocabSize() int32 { return m.Config.VocabSize }

// Tokenizer returns the model's tokenizer
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

// NewCache creates a new KV cache for the model
func (m *Model) NewCache(maxSeqLen int32) []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = cache.NewKVCache()
	}
	return caches
}

// FormatPrompt applies the GLM-4 chat template with thinking enabled by default.
// This follows the GLM-4.7 format with <think> tag for reasoning mode.
func (m *Model) FormatPrompt(prompt string) string {
	return "[gMASK]<sop><|user|>" + prompt + "<|assistant|><think>"
}

// FormatPromptWithThinking applies the GLM-4 chat template with explicit thinking control.
// When think is true, the prompt ends with <think> to enable reasoning mode.
// When think is false, the prompt ends with </think> to skip reasoning.
func (m *Model) FormatPromptWithThinking(prompt string, think bool) string {
	if think {
		return "[gMASK]<sop><|user|>" + prompt + "<|assistant|><think>"
	}
	return "[gMASK]<sop><|user|>" + prompt + "<|assistant|></think>"
}

// NewRenderer returns a new Renderer for formatting multi-turn conversations.
func (m *Model) NewRenderer() *Renderer {
	return &Renderer{}
}

// NewParser returns a new Parser for extracting thinking and tool calls from output.
func (m *Model) NewParser() *Parser {
	return &Parser{}
}
