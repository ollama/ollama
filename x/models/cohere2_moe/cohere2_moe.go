// Package cohere2_moe provides the Cohere2 MoE (Command A family, North) text
// model implementation for MLX.
//
// Architecture notes (matches transformers' Cohere2MoeForCausalLM):
//   - Parallel residual blocks: a single input layernorm feeds both attention
//     and the MLP, and their outputs are summed onto the residual.
//   - Interleaved sliding-window and full attention layers. Sliding layers use
//     interleaved ("traditional") RoPE; full-attention layers use no positional
//     encoding (NoPE), except prefix dense layers when
//     prefix_dense_sliding_window_pattern == 1, which force RoPE.
//   - The first first_k_dense_replace layers use a dense SwiGLU MLP with
//     prefix_dense_intermediate_size; the rest are sparse MoE layers routed by
//     a linear gate with sigmoid or softmax selection over the top-k logits.
//   - Logits are scaled by logit_scale. Embeddings are tied by default.
package cohere2_moe

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("Cohere2MoeForCausalLM", NewModel)
}

// Config holds the Cohere2 MoE configuration (HuggingFace config.json).
type Config struct {
	HiddenSize            int32    `json:"hidden_size"`
	NumHiddenLayers       int32    `json:"num_hidden_layers"`
	IntermediateSize      int32    `json:"intermediate_size"`
	NumAttentionHeads     int32    `json:"num_attention_heads"`
	NumKeyValueHeads      int32    `json:"num_key_value_heads"`
	HeadDim               int32    `json:"head_dim"`
	VocabSize             int32    `json:"vocab_size"`
	MaxPositionEmbeddings int32    `json:"max_position_embeddings"`
	LayerNormEps          float32  `json:"layer_norm_eps"`
	RMSNormEps            *float32 `json:"rms_norm_eps"`
	RopeTheta             float32  `json:"rope_theta"`
	LogitScale            float32  `json:"logit_scale"`
	AttentionBias         bool     `json:"attention_bias"`
	TieWordEmbeddings     *bool    `json:"tie_word_embeddings"`

	SlidingWindow                   int32    `json:"sliding_window"`
	SlidingWindowPattern            int32    `json:"sliding_window_pattern"`
	PrefixDenseSlidingWindowPattern int32    `json:"prefix_dense_sliding_window_pattern"`
	LayerTypes                      []string `json:"layer_types"`
	MLPLayerTypes                   []string `json:"mlp_layer_types"`
	FirstKDenseReplace              int32    `json:"first_k_dense_replace"`
	PrefixDenseIntermediateSize     int32    `json:"prefix_dense_intermediate_size"`

	NumExperts                      int32  `json:"num_experts"`
	NumExpertsPerTok                int32  `json:"num_experts_per_tok"`
	NumSharedExperts                int32  `json:"num_shared_experts"`
	SharedExpertCombinationStrategy string `json:"shared_expert_combination_strategy"`
	ExpertSelectionFn               string `json:"expert_selection_fn"`
	NormTopKProb                    bool   `json:"norm_topk_prob"`

	// Quantization metadata (set at load, not from config.json).
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	Scale float32 `json:"-"`
}

// normLayer abstracts the per-config choice between RMSNorm (rms_norm_eps set)
// and Cohere-style bias-free LayerNorm.
type normLayer interface {
	Forward(x *mlx.Array) *mlx.Array
}

type rmsNorm struct {
	Weight *mlx.Array
	Eps    float32
}

func (n *rmsNorm) Forward(x *mlx.Array) *mlx.Array { return mlx.RMSNormFn(x, n.Weight, n.Eps) }

type layerNorm struct {
	Weight *mlx.Array
	Eps    float32
}

func (n *layerNorm) Forward(x *mlx.Array) *mlx.Array {
	return mlx.LayerNormFn(x, n.Weight, nil, n.Eps)
}

// Model is the Cohere2 MoE model.
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        normLayer
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config
}

// Layer is a parallel-residual transformer block.
type Layer struct {
	InputNorm normLayer
	Attention *Attention
	MLP       MLPBlock

	IsSliding bool
	UseRope   bool
}

// Attention implements Cohere2 attention (no q/k norm).
type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
}

// MLPBlock is the feed-forward interface for dense and MoE blocks.
type MLPBlock interface {
	Forward(x *mlx.Array, cfg *Config) *mlx.Array
}

// DenseMLP is a SwiGLU feed-forward block.
type DenseMLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

// SparseMoE routes each token to the top-k of NumExperts expert MLPs.
type SparseMoE struct {
	Router       nn.LinearLayer
	SwitchMLP    *SwitchMLP
	SharedExpert *DenseMLP
}

// SwitchMLP executes the selected expert MLPs with stacked expert weights.
type SwitchMLP struct {
	GateWeight *mlx.Array
	UpWeight   *mlx.Array
	DownWeight *mlx.Array

	GateWeightQ, GateScales, GateBiases *mlx.Array
	UpWeightQ, UpScales, UpBiases       *mlx.Array
	DownWeightQ, DownScales, DownBiases *mlx.Array

	GateBits, UpBits, DownBits                int
	GateGroupSize, UpGroupSize, DownGroupSize int
	GateMode, UpMode, DownMode                string

	UseQuantized bool
}

type stackedExpertWeights struct {
	Weight    *mlx.Array
	Scales    *mlx.Array
	Biases    *mlx.Array
	Bits      int
	GroupSize int
	Mode      string
}

func parseConfig(configData []byte) (Config, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(configData, &raw); err != nil {
		return Config{}, fmt.Errorf("parse config envelope: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}

	if cfg.HiddenSize <= 0 {
		return Config{}, fmt.Errorf("invalid hidden_size: %d", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers <= 0 {
		return Config{}, fmt.Errorf("invalid num_hidden_layers: %d", cfg.NumHiddenLayers)
	}
	if cfg.NumAttentionHeads <= 0 {
		return Config{}, fmt.Errorf("invalid num_attention_heads: %d", cfg.NumAttentionHeads)
	}
	if cfg.NumKeyValueHeads <= 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HeadDim <= 0 {
		if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
			return Config{}, fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
		}
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}

	// Defaults follow transformers' Cohere2MoeConfig.
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-5
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	if cfg.LogitScale == 0 {
		cfg.LogitScale = 0.0625
	}
	if _, ok := raw["sliding_window"]; !ok {
		cfg.SlidingWindow = 4096
	}
	if cfg.SlidingWindowPattern <= 0 {
		cfg.SlidingWindowPattern = 4
	}
	if cfg.PrefixDenseSlidingWindowPattern <= 0 {
		cfg.PrefixDenseSlidingWindowPattern = 1
	}
	if cfg.MaxPositionEmbeddings <= 0 {
		cfg.MaxPositionEmbeddings = 8192
	}

	if cfg.NumExperts <= 0 {
		cfg.NumExperts = 8
	}
	if cfg.NumExpertsPerTok <= 0 {
		cfg.NumExpertsPerTok = 2
	}
	if cfg.NumExpertsPerTok > cfg.NumExperts {
		return Config{}, fmt.Errorf("num_experts_per_tok (%d) exceeds num_experts (%d)", cfg.NumExpertsPerTok, cfg.NumExperts)
	}
	if cfg.ExpertSelectionFn == "" {
		cfg.ExpertSelectionFn = "softmax"
	}
	if cfg.ExpertSelectionFn != "softmax" && cfg.ExpertSelectionFn != "sigmoid" {
		return Config{}, fmt.Errorf("unsupported expert_selection_fn: %q", cfg.ExpertSelectionFn)
	}
	if cfg.SharedExpertCombinationStrategy == "" {
		cfg.SharedExpertCombinationStrategy = "average"
	}
	if cfg.SharedExpertCombinationStrategy != "average" && cfg.SharedExpertCombinationStrategy != "sum" {
		return Config{}, fmt.Errorf("unsupported shared_expert_combination_strategy: %q", cfg.SharedExpertCombinationStrategy)
	}
	if _, ok := raw["norm_topk_prob"]; !ok {
		cfg.NormTopKProb = true
	}
	if cfg.PrefixDenseIntermediateSize <= 0 {
		cfg.PrefixDenseIntermediateSize = cfg.IntermediateSize
	}

	// Derive per-layer attention types when absent: the first
	// first_k_dense_replace layers follow prefix_dense_sliding_window_pattern,
	// the rest follow sliding_window_pattern (full attention every Nth layer).
	if len(cfg.LayerTypes) == 0 {
		cfg.LayerTypes = make([]string, cfg.NumHiddenLayers)
		for i := range cfg.NumHiddenLayers {
			if i < cfg.FirstKDenseReplace {
				cfg.LayerTypes[i] = patternLayerType(i, cfg.PrefixDenseSlidingWindowPattern)
			} else {
				cfg.LayerTypes[i] = patternLayerType(i-cfg.FirstKDenseReplace, cfg.SlidingWindowPattern)
			}
		}
	}
	if len(cfg.LayerTypes) != int(cfg.NumHiddenLayers) {
		return Config{}, fmt.Errorf("layer_types has %d entries, want %d", len(cfg.LayerTypes), cfg.NumHiddenLayers)
	}

	// Derive per-layer MLP types when absent: the first first_k_dense_replace
	// layers are dense, the rest sparse.
	if len(cfg.MLPLayerTypes) == 0 {
		cfg.MLPLayerTypes = make([]string, cfg.NumHiddenLayers)
		for i := range cfg.NumHiddenLayers {
			if i < cfg.FirstKDenseReplace {
				cfg.MLPLayerTypes[i] = "dense"
			} else {
				cfg.MLPLayerTypes[i] = "sparse"
			}
		}
	}
	if len(cfg.MLPLayerTypes) != int(cfg.NumHiddenLayers) {
		return Config{}, fmt.Errorf("mlp_layer_types has %d entries, want %d", len(cfg.MLPLayerTypes), cfg.NumHiddenLayers)
	}

	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	return cfg, nil
}

func patternLayerType(i, pattern int32) string {
	if pattern > 0 && (i+1)%pattern == 0 {
		return "full_attention"
	}
	return "sliding_attention"
}

func (cfg *Config) layerIsSliding(i int32) bool {
	return cfg.LayerTypes[i] == "sliding_attention"
}

func (cfg *Config) layerIsDense(i int32) bool {
	return cfg.MLPLayerTypes[i] == "dense"
}

// layerUsesRope reports whether layer i applies rotary embeddings: all sliding
// layers do, and prefix dense layers force RoPE even with full attention when
// prefix_dense_sliding_window_pattern == 1 (matching Cohere2MoeAttention's
// force_rope). Other full-attention layers use no positional encoding.
func (cfg *Config) layerUsesRope(i int32) bool {
	if cfg.layerIsSliding(i) {
		return true
	}
	return cfg.layerIsDense(i) && cfg.PrefixDenseSlidingWindowPattern == 1
}

func (cfg *Config) newNorm(weight *mlx.Array) normLayer {
	if cfg.RMSNormEps != nil {
		return &rmsNorm{Weight: weight, Eps: *cfg.RMSNormEps}
	}
	return &layerNorm{Weight: weight, Eps: cfg.LayerNormEps}
}

func (cfg *Config) tieEmbeddings() bool {
	return cfg.TieWordEmbeddings == nil || *cfg.TieWordEmbeddings
}

// NewModel creates a Cohere2 MoE model from a manifest root.
func NewModel(root *model.Root) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	cfg, err := parseConfig(configData)
	if err != nil {
		return nil, err
	}

	if qt := root.QuantType(); qt != "" {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams(qt)
		if gs := root.GroupSize(); gs > 0 {
			cfg.QuantGroupSize = gs
		}
	} else {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams("")
	}
	cfg.TensorQuant = root.AllTensorQuant()

	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer config: %w", err)
	}

	tokConfig := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	if genConfigData, err := root.Manifest.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = genConfigData
	}
	if tokConfigData, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = tokConfigData
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	m := &Model{
		Layers: make([]*Layer, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}

	for i := range cfg.NumHiddenLayers {
		m.Layers[i] = &Layer{
			IsSliding: cfg.layerIsSliding(i),
			UseRope:   cfg.layerUsesRope(i),
		}
	}

	return m, nil
}

func supportsGatherQMM(mode string, bits int) bool {
	switch mode {
	case "affine":
		return bits == 4 || bits == 8
	case "mxfp8":
		return bits == 8
	case "nvfp4", "mxfp4":
		return bits == 4
	default:
		return false
	}
}

// transposeExpertWeightForGatherMM converts stacked [E, out, in] expert
// weights to the [E, in, out] layout GatherMM consumes, materialized once at
// load so the forward path avoids per-call transposes.
func transposeExpertWeightForGatherMM(w *mlx.Array) *mlx.Array {
	if w == nil || !w.Valid() || w.NumDims() != 3 {
		return w
	}
	t := mlx.Transpose(w, 0, 2, 1)
	cloned := t.Clone()
	mlx.Eval(cloned)
	return cloned
}

// loadStackedProjection returns expert weights already stacked as a single 3D
// tensor (layers.N.mlp.switch_mlp.<proj>.weight) — the layout `ollama create`
// writes when it packs per-expert tensors at import.
func loadStackedProjection(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, base string) *stackedExpertWeights {
	key := base + ".weight"
	w := tensors[key]
	if w == nil {
		return nil
	}

	scales := tensors[key+"_scale"]
	if scales == nil {
		return &stackedExpertWeights{Weight: w}
	}

	qbiases := tensors[key+"_qbias"]
	groupSize, bits, mode := model.ResolveLinearQuantParams(
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant,
		key, w, scales,
	)
	if useQuantized && supportsGatherQMM(mode, bits) {
		return &stackedExpertWeights{
			Weight:    w,
			Scales:    scales,
			Biases:    qbiases,
			Bits:      bits,
			GroupSize: groupSize,
			Mode:      mode,
		}
	}

	return &stackedExpertWeights{
		Weight:    mlx.Dequantize(w, scales, qbiases, groupSize, bits, mode),
		Bits:      bits,
		GroupSize: groupSize,
		Mode:      mode,
	}
}

// loadStackedExperts resolves a stacked expert projection by its close-to-source
// name (layers.N.mlp.experts.<proj>, which `ollama create` now writes) and falls
// back to the legacy switch_mlp name that older imports produced.
func loadStackedExperts(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, layerPrefix, proj string) *stackedExpertWeights {
	if w := loadStackedProjection(tensors, cfg, useQuantized, layerPrefix+".mlp.experts."+proj); w != nil {
		return w
	}
	return loadStackedProjection(tensors, cfg, useQuantized, layerPrefix+".mlp.switch_mlp."+proj)
}

// LoadWeights assigns tensors to model fields.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	cfg := m.Config

	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)

	embedTokens := model.MakeEmbeddingLayer(tensors, "model.embed_tokens", cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	if embedTokens == nil {
		return fmt.Errorf("missing embedding weight: model.embed_tokens.weight")
	}
	m.EmbedTokens = embedTokens

	normWeight := tensors["model.norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: model.norm.weight")
	}
	m.Norm = cfg.newNorm(normWeight)

	if cfg.tieEmbeddings() {
		m.LMHead = m.EmbedTokens.AsLinear()
	} else if lmHead := linears.Make("lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else {
		m.LMHead = m.EmbedTokens.AsLinear()
	}

	useQuantizedExperts := supportsGatherQMM(cfg.QuantMode, cfg.QuantBits)
	if !useQuantizedExperts && cfg.TensorQuant != nil {
		for _, tq := range cfg.TensorQuant {
			if tq == nil {
				continue
			}
			_, bits, mode := model.QuantizationParams(tq.QuantType)
			if supportsGatherQMM(mode, bits) {
				useQuantizedExperts = true
				break
			}
		}
	}

	for i := range cfg.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("model.layers.%d", i)
		layer := &Layer{
			IsSliding: cfg.layerIsSliding(i),
			UseRope:   cfg.layerUsesRope(i),
		}

		normWeight := tensors[layerPrefix+".input_layernorm.weight"]
		if normWeight == nil {
			return fmt.Errorf("layer %d: missing input_layernorm", i)
		}
		layer.InputNorm = cfg.newNorm(normWeight)

		attn := &Attention{
			QProj: linears.Make(layerPrefix + ".self_attn.q_proj"),
			KProj: linears.Make(layerPrefix + ".self_attn.k_proj"),
			VProj: linears.Make(layerPrefix + ".self_attn.v_proj"),
			OProj: linears.Make(layerPrefix + ".self_attn.o_proj"),
		}
		if attn.QProj == nil || attn.KProj == nil || attn.VProj == nil || attn.OProj == nil {
			return fmt.Errorf("layer %d: missing attention projections", i)
		}
		layer.Attention = attn

		if cfg.layerIsDense(i) {
			mlp := &DenseMLP{
				GateProj: linears.Make(layerPrefix + ".mlp.gate_proj"),
				UpProj:   linears.Make(layerPrefix + ".mlp.up_proj"),
				DownProj: linears.Make(layerPrefix + ".mlp.down_proj"),
			}
			if mlp.GateProj == nil || mlp.UpProj == nil || mlp.DownProj == nil {
				return fmt.Errorf("layer %d: missing dense mlp projections", i)
			}
			layer.MLP = mlp
		} else {
			moe := &SparseMoE{}
			moe.Router = linears.Make(layerPrefix + ".mlp.gate")
			if moe.Router == nil {
				return fmt.Errorf("layer %d: missing moe router gate", i)
			}

			gateW := loadStackedExperts(tensors, cfg, useQuantizedExperts, layerPrefix, "gate_proj")
			upW := loadStackedExperts(tensors, cfg, useQuantizedExperts, layerPrefix, "up_proj")
			downW := loadStackedExperts(tensors, cfg, useQuantizedExperts, layerPrefix, "down_proj")
			if gateW == nil || upW == nil || downW == nil {
				return fmt.Errorf("layer %d: missing stacked expert weights (import the model with `ollama create`)", i)
			}

			switchMLP := &SwitchMLP{}
			if gateW.Scales != nil && upW.Scales != nil && downW.Scales != nil {
				switchMLP.UseQuantized = true
				switchMLP.GateWeightQ = gateW.Weight
				switchMLP.GateScales = gateW.Scales
				switchMLP.GateBiases = gateW.Biases
				switchMLP.GateBits = gateW.Bits
				switchMLP.GateGroupSize = gateW.GroupSize
				switchMLP.GateMode = gateW.Mode
				switchMLP.UpWeightQ = upW.Weight
				switchMLP.UpScales = upW.Scales
				switchMLP.UpBiases = upW.Biases
				switchMLP.UpBits = upW.Bits
				switchMLP.UpGroupSize = upW.GroupSize
				switchMLP.UpMode = upW.Mode
				switchMLP.DownWeightQ = downW.Weight
				switchMLP.DownScales = downW.Scales
				switchMLP.DownBiases = downW.Biases
				switchMLP.DownBits = downW.Bits
				switchMLP.DownGroupSize = downW.GroupSize
				switchMLP.DownMode = downW.Mode
			} else {
				switchMLP.GateWeight = transposeExpertWeightForGatherMM(gateW.Weight)
				switchMLP.UpWeight = transposeExpertWeightForGatherMM(upW.Weight)
				switchMLP.DownWeight = transposeExpertWeightForGatherMM(downW.Weight)
			}
			moe.SwitchMLP = switchMLP

			if cfg.NumSharedExperts > 0 {
				shared := &DenseMLP{
					GateProj: linears.Make(layerPrefix + ".mlp.shared_experts.gate_proj"),
					UpProj:   linears.Make(layerPrefix + ".mlp.shared_experts.up_proj"),
					DownProj: linears.Make(layerPrefix + ".mlp.shared_experts.down_proj"),
				}
				if shared.GateProj == nil {
					shared.GateProj = linears.Make(layerPrefix + ".mlp.shared_expert.gate_proj")
					shared.UpProj = linears.Make(layerPrefix + ".mlp.shared_expert.up_proj")
					shared.DownProj = linears.Make(layerPrefix + ".mlp.shared_expert.down_proj")
				}
				if shared.GateProj == nil || shared.UpProj == nil || shared.DownProj == nil {
					return fmt.Errorf("layer %d: missing shared expert projections", i)
				}
				moe.SharedExpert = shared
			}

			layer.MLP = moe
		}

		m.Layers[i] = layer
	}

	return nil
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, useRope bool, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Transpose(mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim), 0, 2, 1, 3)
	k = mlx.Transpose(mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim), 0, 2, 1, 3)
	v = mlx.Transpose(mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim), 0, 2, 1, 3)

	// Cohere uses interleaved pairs (traditional RoPE). Full-attention layers
	// outside the forced-RoPE prefix use no positional encoding.
	if useRope {
		q = mlx.RoPEWithBase(q, int(cfg.HeadDim), true, cfg.RopeTheta, 1.0, positions)
		k = mlx.RoPEWithBase(k, int(cfg.HeadDim), true, cfg.RopeTheta, 1.0, positions)
	}

	var kv nn.SDPAOption
	if c != nil {
		history := c.(cache.Attention).Update(b, k, v)
		kv = nn.WithKVHistory(history)
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
	}
	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(nn.CausalMask()))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (m *DenseMLP) Forward(x *mlx.Array, _ *Config) *mlx.Array {
	return m.DownProj.Forward(mlx.SwiGLU(m.GateProj.Forward(x), m.UpProj.Forward(x)))
}

// route selects the top-k experts. Selection happens on the raw router logits
// and the activation (sigmoid or softmax) is applied to just the selected
// entries, matching Cohere2MoeTopKRouter (both activations are monotonic, so
// selection order is unchanged).
func (moe *SparseMoE) route(x *mlx.Array, cfg *Config) (inds, scores *mlx.Array) {
	logits := moe.Router.Forward(x)

	inds = mlx.Argpartition(mlx.Neg(logits), int(cfg.NumExpertsPerTok)-1, -1)
	dims := inds.Dims()
	inds = mlx.SliceStartStop(inds, []int32{0, 0, 0}, []int32{int32(dims[0]), int32(dims[1]), cfg.NumExpertsPerTok})

	selected := mlx.TakeAlongAxis(logits, inds, -1)
	if cfg.ExpertSelectionFn == "sigmoid" {
		scores = mlx.Sigmoid(selected)
		if cfg.NormTopKProb && cfg.NumExpertsPerTok > 1 {
			scores = mlx.Div(scores, mlx.Sum(scores, -1, true))
		}
	} else {
		scores = mlx.SoftmaxAxis(selected, -1, true)
	}
	return inds, scores
}

func (moe *SparseMoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	inds, scores := moe.route(x, cfg)

	expertOut := moe.SwitchMLP.Forward(x, inds, cfg)
	y := mlx.Sum(mlx.Mul(expertOut, mlx.ExpandDims(scores, -1)), 2, false)

	if moe.SharedExpert != nil {
		y = mlx.Add(y, moe.SharedExpert.Forward(x, cfg))
		if cfg.SharedExpertCombinationStrategy == "average" {
			y = mlx.MulScalar(y, 0.5)
		}
	}

	return mlx.Reshape(y, B, L, cfg.HiddenSize)
}

func (s *SwitchMLP) Forward(x *mlx.Array, indices *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := cfg.NumExpertsPerTok

	xFlat := mlx.Reshape(x, B*L, 1, 1, cfg.HiddenSize)
	idxFlat := mlx.Reshape(indices, B*L, topK)

	// Sorting tokens by expert improves gather matmul locality for prefill
	// batches; the cost outweighs the benefit for small decode batches.
	doSort := B*L >= 64
	var invOrder *mlx.Array
	n := B * L * topK

	if doSort {
		idxAll := mlx.Flatten(idxFlat)
		order := mlx.Argsort(idxAll, 0)
		invOrder = mlx.Argsort(order, 0)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), mlx.FloorDivideScalar(order, topK), 0), 1)
		idxFlat = mlx.Reshape(mlx.Take(idxAll, order, 0), n, 1)
	}

	var gate, up, hidden, down *mlx.Array
	if s.UseQuantized {
		gate = mlx.GatherQMM(xFlat, s.GateWeightQ, s.GateScales, s.GateBiases,
			nil, idxFlat, true, s.GateGroupSize, s.GateBits, s.GateMode, doSort)
		up = mlx.GatherQMM(xFlat, s.UpWeightQ, s.UpScales, s.UpBiases,
			nil, idxFlat, true, s.UpGroupSize, s.UpBits, s.UpMode, doSort)
		hidden = mlx.SwiGLU(gate, up)
		down = mlx.GatherQMM(hidden, s.DownWeightQ, s.DownScales, s.DownBiases,
			nil, idxFlat, true, s.DownGroupSize, s.DownBits, s.DownMode, doSort)
	} else {
		gate = mlx.GatherMM(xFlat, s.GateWeight, nil, idxFlat, doSort)
		up = mlx.GatherMM(xFlat, s.UpWeight, nil, idxFlat, doSort)
		hidden = mlx.SwiGLU(gate, up)
		down = mlx.GatherMM(hidden, s.DownWeight, nil, idxFlat, doSort)
	}

	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

// Forward runs a parallel-residual block: one shared layernorm feeds both
// attention and the MLP, and the residual adds both outputs.
func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	normed := l.InputNorm.Forward(x)
	attnOut := l.Attention.Forward(normed, b, c, positions, B, L, l.UseRope, cfg)
	mlpOut := l.MLP.Forward(normed, cfg)
	return mlx.Add(x, mlx.Add(attnOut, mlpOut))
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))

	h := m.EmbedTokens.Forward(b.InputIDs)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
	}

	return m.Norm.Forward(h)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	logits := m.LMHead.Forward(x)
	if m.LogitScale != 1.0 {
		logits = mlx.MulScalar(logits, m.LogitScale)
	}
	return logits
}

func (m *Model) NumLayers() int {
	return len(m.Layers)
}

func (m *Model) MaxContextLength() int {
	return int(m.MaxPositionEmbeddings)
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.tok
}

// NewCaches creates per-layer caches: rotating (bounded) caches for sliding
// window layers and standard KV caches for full attention layers.
func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i, layer := range m.Layers {
		if m.SlidingWindow > 0 && layer.IsSliding {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}
