//go:build mlx

// Package glm4_moe_lite provides the GLM4-MoE-Lite implementation for MLX.
// This model uses Multi-head Latent Attention (MLA) and Mixture of Experts (MoE).
package glm4_moe_lite

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
	"github.com/ollama/ollama/x/imagegen/safetensors"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

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
	QLoraRank      int32 `json:"q_lora_rank"`
	KVLoraRank     int32 `json:"kv_lora_rank"`
	QKRopeHeadDim  int32 `json:"qk_rope_head_dim"`
	QKNopeHeadDim  int32 `json:"qk_nope_head_dim"`
	VHeadDim       int32 `json:"v_head_dim"`

	// MoE parameters
	NRoutedExperts      int32   `json:"n_routed_experts"`
	NSharedExperts      int32   `json:"n_shared_experts"`
	NumExpertsPerTok    int32   `json:"num_experts_per_tok"`
	RoutedScalingFactor float32 `json:"routed_scaling_factor"`
	NormTopKProb        bool    `json:"norm_topk_prob"`
	FirstKDenseReplace  int32   `json:"first_k_dense_replace"`
	NGroup              int32   `json:"n_group"`
	TopKGroup           int32   `json:"topk_group"`

	// Computed fields
	QHeadDim int32   `json:"-"` // qk_nope_head_dim + qk_rope_head_dim
	Scale    float32 `json:"-"` // 1/sqrt(QHeadDim)
}

// MLAAttention implements Multi-head Latent Attention
type MLAAttention struct {
	// Low-rank query projections
	QAProj      *nn.Linear  `weight:"self_attn.q_a_proj"`
	QALayerNorm *nn.RMSNorm `weight:"self_attn.q_a_layernorm"`
	QBProj      *nn.Linear  `weight:"self_attn.q_b_proj"`

	// Low-rank KV projections (with shared rope component)
	KVAProjWithMQA *nn.Linear  `weight:"self_attn.kv_a_proj_with_mqa"`
	KVALayerNorm   *nn.RMSNorm `weight:"self_attn.kv_a_layernorm"`
	KVBProj        *nn.Linear  `weight:"self_attn.kv_b_proj"`

	// Output projection
	OProj *nn.Linear `weight:"self_attn.o_proj"`
}

// Forward computes MLA attention output
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

	// KV path: kv_a_proj_with_mqa -> split -> layernorm -> kv_b_proj
	compressedKV := a.KVAProjWithMQA.Forward(x)

	// Split into compressed_kv and k_pe (shared rope component)
	kvCompressed := mlx.Slice(compressedKV, []int32{0, 0, 0}, []int32{B, L, cfg.KVLoraRank})
	kPE := mlx.Slice(compressedKV, []int32{0, 0, cfg.KVLoraRank}, []int32{B, L, cfg.KVLoraRank + cfg.QKRopeHeadDim})

	// k_pe is shared across heads (MQA-style): [B, L, rope_dim] -> [B, 1, L, rope_dim]
	kPE = mlx.Reshape(kPE, B, L, 1, cfg.QKRopeHeadDim)
	kPE = mlx.Transpose(kPE, 0, 2, 1, 3)

	// Apply layernorm and project KV
	kvCompressed = a.KVALayerNorm.Forward(kvCompressed, cfg.RMSNormEps)
	kv := a.KVBProj.Forward(kvCompressed)

	// Reshape KV: [B, L, num_heads * (qk_nope_head_dim + v_head_dim)]
	kv = mlx.Reshape(kv, B, L, cfg.NumAttentionHeads, cfg.QKNopeHeadDim+cfg.VHeadDim)
	kv = mlx.Transpose(kv, 0, 2, 1, 3)

	// Split into k_nope and values
	kNope := mlx.Slice(kv, []int32{0, 0, 0, 0}, []int32{B, cfg.NumAttentionHeads, L, cfg.QKNopeHeadDim})
	values := mlx.Slice(kv, []int32{0, 0, 0, cfg.QKNopeHeadDim}, []int32{B, cfg.NumAttentionHeads, L, cfg.QKNopeHeadDim + cfg.VHeadDim})

	// Apply RoPE to the rope parts only
	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	qPE = mlx.RoPE(qPE, int(cfg.QKRopeHeadDim), true, cfg.RopeTheta, 1.0, offset)
	kPE = mlx.RoPE(kPE, int(cfg.QKRopeHeadDim), true, cfg.RopeTheta, 1.0, offset)

	// Repeat k_pe across all heads
	kPE = mlx.Tile(kPE, []int32{1, cfg.NumAttentionHeads, 1, 1})

	// Concatenate nope and rope parts
	queries := mlx.Concatenate([]*mlx.Array{qNope, qPE}, 3)
	keys := mlx.Concatenate([]*mlx.Array{kNope, kPE}, 3)

	// Update KV cache
	if c != nil {
		keys, values = c.Update(keys, values, int(L))
	}

	// Scaled dot product attention
	out := mlx.ScaledDotProductAttention(queries, keys, values, cfg.Scale, L > 1)

	// Reshape back: [B, num_heads, L, v_head_dim] -> [B, L, num_heads * v_head_dim]
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.VHeadDim)

	return a.OProj.Forward(out)
}

// DenseMLP implements the standard SwiGLU MLP for dense layers
type DenseMLP struct {
	GateProj *nn.Linear `weight:"mlp.gate_proj"`
	UpProj   *nn.Linear `weight:"mlp.up_proj"`
	DownProj *nn.Linear `weight:"mlp.down_proj"`
}

// Forward applies the SwiGLU MLP
func (m *DenseMLP) Forward(x *mlx.Array) *mlx.Array {
	gate := mlx.SiLU(m.GateProj.Forward(x))
	up := m.UpProj.Forward(x)
	return m.DownProj.Forward(mlx.Mul(gate, up))
}

// MoEGate implements the expert gating mechanism
type MoEGate struct {
	Weight                 *mlx.Array `weight:"mlp.gate.weight"`
	EScoreCorrectionBias   *mlx.Array `weight:"mlp.gate.e_score_correction_bias,optional"`
}

// Forward computes expert selection indices and scores
func (g *MoEGate) Forward(x *mlx.Array, cfg *Config) (*mlx.Array, *mlx.Array) {
	// Compute gate logits: x @ weight.T
	gates := mlx.Linear(x, mlx.Transpose(g.Weight, 1, 0))

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
	GateWeight *mlx.Array
	UpWeight   *mlx.Array
	DownWeight *mlx.Array
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

	// Expert computation using gather_mm
	// gate: x @ gate_weight.T (indices are on the rhs/weight side)
	gate := mlx.GatherMM(xFlat, mlx.Transpose(s.GateWeight, 0, 2, 1), nil, idxFlat, doSort)
	// up: x @ up_weight.T
	up := mlx.GatherMM(xFlat, mlx.Transpose(s.UpWeight, 0, 2, 1), nil, idxFlat, doSort)

	// SwiGLU activation
	hidden := mlx.Mul(mlx.SiLU(gate), up)

	// down: hidden @ down_weight.T
	down := mlx.GatherMM(hidden, mlx.Transpose(s.DownWeight, 0, 2, 1), nil, idxFlat, doSort)

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
	GateProj *nn.Linear `weight:"mlp.shared_experts.gate_proj"`
	UpProj   *nn.Linear `weight:"mlp.shared_experts.up_proj"`
	DownProj *nn.Linear `weight:"mlp.shared_experts.down_proj"`
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
	EmbedTokens *nn.Embedding `weight:"model.embed_tokens"`
	Layers      []Block       `weight:"-"` // Loaded manually due to different block types
	Norm        *nn.RMSNorm   `weight:"model.norm"`
	LMHead      *nn.Linear    `weight:"lm_head"`

	tok *tokenizer.Tokenizer
	*Config
}

// sanitizeExpertWeights stacks individual expert weights into a single tensor
func sanitizeExpertWeights(weights *safetensors.ModelWeights, prefix string, numExperts int32) (*mlx.Array, *mlx.Array, *mlx.Array) {
	var gateWeights, upWeights, downWeights []*mlx.Array

	for e := int32(0); e < numExperts; e++ {
		gw, _ := weights.GetTensor(fmt.Sprintf("%s.mlp.experts.%d.gate_proj.weight", prefix, e))
		uw, _ := weights.GetTensor(fmt.Sprintf("%s.mlp.experts.%d.up_proj.weight", prefix, e))
		dw, _ := weights.GetTensor(fmt.Sprintf("%s.mlp.experts.%d.down_proj.weight", prefix, e))

		if gw != nil {
			gateWeights = append(gateWeights, gw)
		}
		if uw != nil {
			upWeights = append(upWeights, uw)
		}
		if dw != nil {
			downWeights = append(downWeights, dw)
		}
	}

	var stackedGate, stackedUp, stackedDown *mlx.Array
	if len(gateWeights) > 0 {
		stackedGate = mlx.Stack(gateWeights, 0)
	}
	if len(upWeights) > 0 {
		stackedUp = mlx.Stack(upWeights, 0)
	}
	if len(downWeights) > 0 {
		stackedDown = mlx.Stack(downWeights, 0)
	}

	return stackedGate, stackedUp, stackedDown
}

// Load loads a GLM4-MoE-Lite model from the given path
func Load(modelPath string) (*Model, error) {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Compute derived fields
	cfg.QHeadDim = cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.QHeadDim)))

	weights, err := safetensors.LoadModelWeights(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	tok, err := tokenizer.Load(filepath.Join(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
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

			// Stack expert weights
			gateW, upW, downW := sanitizeExpertWeights(weights, prefix, cfg.NRoutedExperts)

			block.MoE = &MoE{
				Gate: &MoEGate{},
				SwitchMLP: &SwitchMLP{
					GateWeight: gateW,
					UpWeight:   upW,
					DownWeight: downW,
				},
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

// FormatPrompt applies the GLM-4 chat template
func (m *Model) FormatPrompt(prompt string) string {
	return "[gMASK]<sop><|user|>\n" + prompt + "<|assistant|>\n"
}
