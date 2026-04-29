// Package gemma4 provides the Gemma 4 text model implementation for MLX.
package gemma4

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
	base.Register("Gemma4ForCausalLM", newModel)
	base.Register("Gemma4ForConditionalGeneration", newModel)
}

// Compile-time interface checks.
var _ base.Model = (*Model)(nil)

// RopeParams holds per-layer-type RoPE settings.
type RopeParams struct {
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	RopeTheta           float32 `json:"rope_theta"`
	RopeType            string  `json:"rope_type"`
}

// TextConfig holds configuration for the Gemma 4 text model.
type TextConfig struct {
	HiddenSize             int32                  `json:"hidden_size"`
	NumHiddenLayers        int32                  `json:"num_hidden_layers"`
	IntermediateSize       int32                  `json:"intermediate_size"`
	NumAttentionHeads      int32                  `json:"num_attention_heads"`
	NumKeyValueHeads       int32                  `json:"num_key_value_heads"`
	HeadDim                int32                  `json:"head_dim"`
	GlobalHeadDim          int32                  `json:"global_head_dim"`
	VocabSize              int32                  `json:"vocab_size"`
	RMSNormEps             float32                `json:"rms_norm_eps"`
	MaxPositionEmbeddings  int32                  `json:"max_position_embeddings"`
	SlidingWindow          int32                  `json:"sliding_window"`
	SlidingWindowPattern   int32                  `json:"sliding_window_pattern"`
	LayerTypes             []string               `json:"layer_types"`
	TieWordEmbeddings      bool                   `json:"tie_word_embeddings"`
	FinalLogitSoftcapping  float32                `json:"final_logit_softcapping"`
	UseDoubleWideMLP       bool                   `json:"use_double_wide_mlp"`
	NumKVSharedLayers      int32                  `json:"num_kv_shared_layers"`
	HiddenSizePerLayer     int32                  `json:"hidden_size_per_layer_input"`
	VocabSizePerLayer      int32                  `json:"vocab_size_per_layer_input"`
	AttentionKEqV          bool                   `json:"attention_k_eq_v"`
	NumGlobalKeyValueHeads int32                  `json:"num_global_key_value_heads"`
	EnableMoeBlock         bool                   `json:"enable_moe_block"`
	NumExperts             int32                  `json:"num_experts"`
	TopKExperts            int32                  `json:"top_k_experts"`
	ExpertIntermediateSize int32                  `json:"moe_intermediate_size"`
	RopeParameters         map[string]*RopeParams `json:"rope_parameters"`
	ImageTokenIDValue      int32                  `json:"image_token_id"`

	// Quantization parameters.
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	SlidingScale    float32    `json:"-"` // 1/sqrt(HeadDim) for sliding layers
	FullScale       float32    `json:"-"` // 1/sqrt(GlobalHeadDim) for full layers
	SlidingRopeDims int        `json:"-"` // HeadDim (full rotation for sliding)
	FullRopeDims    int        `json:"-"` // GlobalHeadDim (partial rotation via custom freqs)
	SlidingRopeBase float32    `json:"-"`
	FullRopeBase    float32    `json:"-"`
	FullRopeFreqs   *mlx.Array `json:"-"` // Precomputed proportional RoPE frequencies

	// Precomputed scale factors (avoid per-forward math.Sqrt/Pow).
	EmbedScale      float32 `json:"-"` // sqrt(hidden_size)
	PLEScale        float32 `json:"-"` // sqrt(hidden_size_per_layer_input)
	PLEProjScale    float32 `json:"-"` // 1/sqrt(hidden_size)
	PLECombineScale float32 `json:"-"` // 2^(-0.5) = 0.7071...
	RouterScale     float32 `json:"-"` // 1/sqrt(hidden_size)

	// KV sharing: maps shared layer index -> donor layer index.
	KVShareMap map[int32]int32 `json:"-"`
	// Set of donor layer indices that need to store their KV.
	KVDonors map[int32]bool `json:"-"`
}

// sharedHistory carries a donor layer's K/V to donees that share
// it. Exactly one of history or (k, v) is populated: history when
// the donor had a cache (donees feed it to SDPA via WithKVHistory;
// (k, v) when the donor had no cache (donees feed them via WithKV).
// mask carries any storage-side mask the (k, v) path needs (e.g.
// sliding window) — unused on the history path, where the cache's
// applier supplies the equivalent restriction.
type sharedHistory struct {
	history *nn.KVHistory
	k, v    *mlx.Array
	mask    nn.AttentionMask
}

// Attention implements Gemma 4 attention with Q/K normalization and v-norm.
type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer

	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm

	// Norm weight for Q/K RMSNorm.
	QNormScaled *mlx.Array
	KNormScaled *mlx.Array
}

// MLP is the feed-forward network with GELU activation.
type MLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

// stackedExpertResult holds the result of collecting and stacking per-expert weights.
type stackedExpertResult struct {
	Weight    *mlx.Array
	Scales    *mlx.Array
	Biases    *mlx.Array
	Bits      int
	GroupSize int
	Mode      string
}

// firstNonNil returns the first non-nil tensor found under any of the given keys.
func firstNonNil(tensors map[string]*mlx.Array, keys ...string) *mlx.Array {
	for _, k := range keys {
		if t := tensors[k]; t != nil {
			return t
		}
	}
	return nil
}

// sliceAxis1 slices a tensor along axis 1: a[:, start:stop, ...].
func sliceAxis1(a *mlx.Array, start, stop int32) *mlx.Array {
	dims := a.Dims()
	beg := make([]int32, len(dims))
	end := make([]int32, len(dims))
	for i, d := range dims {
		end[i] = int32(d)
	}
	beg[1] = start
	end[1] = stop
	return mlx.SliceStartStop(a, beg, end)
}

// transposeForGatherMM transposes stacked expert weights from [experts, out, in]
// to [experts, in, out] for use with GatherMM (which computes a @ b[group]).
func transposeForGatherMM(w *mlx.Array) *mlx.Array {
	if w == nil || !w.Valid() || w.NumDims() != 3 {
		return w
	}
	t := mlx.Transpose(w, 0, 2, 1).Clone()
	mlx.Eval(t)
	return t
}

// collectExpertProjection collects per-expert tensors, stacks them, and
// optionally keeps quantized weight/scale/bias for GatherQMM.
// prefix: e.g. "model.language_model.layers.0.moe.experts"
// proj: e.g. "gate_proj"
func collectExpertProjection(tensors map[string]*mlx.Array, cfg *TextConfig, prefix, proj string, numExperts int32) *stackedExpertResult {
	weights := make([]*mlx.Array, 0, numExperts)
	scales := make([]*mlx.Array, 0, numExperts)
	biases := make([]*mlx.Array, 0, numExperts)
	bits, groupSize := 0, 0
	mode := cfg.QuantMode

	for e := range numExperts {
		// Try "prefix.E.proj.weight" then "prefix.E.proj"
		base := fmt.Sprintf("%s.%d.%s", prefix, e, proj)
		w := tensors[base+".weight"]
		key := base + ".weight"
		if w == nil {
			w = tensors[base]
			key = base
		}
		if w == nil {
			return nil
		}

		s := tensors[key+"_scale"]
		if s == nil {
			weights = append(weights, w)
			continue
		}
		qb := tensors[key+"_qbias"]
		gs, b, m := model.ResolveLinearQuantParams(
			cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode,
			cfg.TensorQuant, key, w, s,
		)
		if bits == 0 {
			bits = b
			groupSize = gs
			mode = m
		}
		// Keep quantized weights for GatherQMM (supports affine, nvfp4, mxfp8).
		weights = append(weights, w)
		scales = append(scales, s)
		if qb != nil {
			biases = append(biases, qb)
		}
	}

	if len(weights) == 0 {
		return nil
	}

	stacked := mlx.Stack(weights, 0).Clone()
	mlx.Eval(stacked)
	out := &stackedExpertResult{Weight: stacked, Bits: bits, GroupSize: groupSize, Mode: mode}
	if len(scales) == len(weights) {
		out.Scales = mlx.Stack(scales, 0).Clone()
		mlx.Eval(out.Scales)
	}
	if len(biases) == len(weights) {
		out.Biases = mlx.Stack(biases, 0).Clone()
		mlx.Eval(out.Biases)
	}
	return out
}

// Router implements Gemma 4's expert routing mechanism.
type Router struct {
	Proj  nn.LinearLayer // [hidden_size -> num_experts]
	Scale *mlx.Array     // learnable scale [hidden_size]
}

// MoEBlock implements the Gemma 4 mixture-of-experts block.
// Uses GatherQMM for quantized weights, GatherMM for dense.
type MoEBlock struct {
	// Dense expert weights for GatherMM (used when not quantized).
	GateUpWeight *mlx.Array // [num_experts, 2*intermediate, hidden] (fused gate+up)
	GateWeight   *mlx.Array // [num_experts, hidden_size, expert_intermediate_size]
	UpWeight     *mlx.Array // [num_experts, hidden_size, expert_intermediate_size]
	DownWeight   *mlx.Array // [num_experts, expert_intermediate_size, hidden_size]

	// Quantized expert weights for GatherQMM.
	GateUpWeightQ, GateUpScales, GateUpBiases *mlx.Array // fused gate+up
	GateWeightQ, GateScales, GateBiases       *mlx.Array
	UpWeightQ, UpScales, UpBiases             *mlx.Array
	DownWeightQ, DownScales, DownBiases       *mlx.Array

	PerExpertScale *mlx.Array // [num_experts]
	UseQuantized   bool
	UseFusedGateUp bool // true when gate+up are stored as single tensor

	// Per-projection quant params (may differ due to mixed-precision).
	GateUpGroupSize, GateUpBits int
	GateGroupSize, UpGroupSize  int
	DownGroupSize               int
	GateBits, UpBits, DownBits  int
	QuantMode                   string // gate/up mode
	DownQuantMode               string // down mode (may differ for mixed mxfp4/mxfp8)
}

// PLELayer holds per-layer PLE weights for a single decoder layer.
type PLELayer struct {
	InputGate  nn.LinearLayer // [hidden_size -> ple_dim]
	Projection nn.LinearLayer // [ple_dim -> hidden_size]
	PostNorm   *nn.RMSNorm

	// Norm weight for post-norm.
	PostNormScaled *mlx.Array
}

// DecoderLayer is a single transformer block.
type DecoderLayer struct {
	InputNorm    *nn.RMSNorm
	Attention    *Attention
	PostAttnNorm *nn.RMSNorm
	PreFFNorm    *nn.RMSNorm
	MLP          *MLP
	PostFFNorm   *nn.RMSNorm

	// PLE per-layer components (nil if no PLE).
	PLE *PLELayer

	// MoE components (nil if no MoE).
	Router *Router
	MoE    *MoEBlock

	// Additional norms for MoE dual-path (nil if no MoE).
	PostFFNorm1 *nn.RMSNorm // post-norm for dense MLP path
	PostFFNorm2 *nn.RMSNorm // post-norm for MoE path
	PreFFNorm2  *nn.RMSNorm // pre-norm for MoE input

	// Norm weight for RMSNorm.
	InputNormScaled    *mlx.Array
	PostAttnNormScaled *mlx.Array
	PreFFNormScaled    *mlx.Array
	PostFFNormScaled   *mlx.Array

	// Norm weight for MoE norms.
	PostFFNorm1Scaled *mlx.Array
	PostFFNorm2Scaled *mlx.Array
	PreFFNorm2Scaled  *mlx.Array

	// Layer scalar for full-attention layers (nil for sliding).
	LayerScalar *mlx.Array

	// Layer metadata.
	IsSliding    bool
	LayerIdx     int32
	KVShareDonor int32 // -1 if not shared, else index of donor layer
	IsDonor      bool  // true if this layer's KV is shared by later layers
}

// Model is the Gemma 4 model (text + optional vision).
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*DecoderLayer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	// PLE model-level components (nil if no PLE).
	EmbedTokensPerLayer nn.EmbeddingLayer
	PerLayerModelProj   nn.LinearLayer
	PerLayerProjNorm    *nn.RMSNorm

	// Precomputed scaled weights.
	NormScaled             *mlx.Array
	PerLayerProjNormWeight *mlx.Array

	tok *tokenizer.Tokenizer
	*TextConfig

	weightPrefix string
}

func parseTextConfig(configData []byte) (TextConfig, error) {
	var cfg TextConfig
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return TextConfig{}, fmt.Errorf("parse config: %w", err)
	}

	var wrapped struct {
		TextConfig *TextConfig `json:"text_config"`
	}
	if err := json.Unmarshal(configData, &wrapped); err != nil {
		return TextConfig{}, fmt.Errorf("parse nested text config: %w", err)
	}

	if wrapped.TextConfig != nil {
		cfg = *wrapped.TextConfig
	}

	// Apply defaults.
	if cfg.HeadDim == 0 {
		cfg.HeadDim = 256
	}
	if cfg.GlobalHeadDim == 0 {
		cfg.GlobalHeadDim = cfg.HeadDim
	}
	if cfg.NumAttentionHeads == 0 {
		cfg.NumAttentionHeads = 8
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = 1
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 262144
	}
	if cfg.SlidingWindowPattern <= 0 && len(cfg.LayerTypes) == 0 {
		cfg.SlidingWindowPattern = 5
	}
	if cfg.MaxPositionEmbeddings == 0 {
		cfg.MaxPositionEmbeddings = 131072
	}

	// Gemma 4 uses scaling=1.0 (no 1/sqrt(head_dim) scaling); the Q/K norms
	// handle magnitude control. This differs from Gemma 3 which uses
	// query_pre_attn_scalar^(-0.5).
	cfg.SlidingScale = 1.0
	cfg.FullScale = 1.0

	// Compute RoPE settings from rope_parameters.
	cfg.SlidingRopeDims = int(cfg.HeadDim) // full rotation for sliding
	cfg.SlidingRopeBase = 10000
	cfg.FullRopeDims = int(cfg.HeadDim) // default: full rotation
	cfg.FullRopeBase = 1000000

	if rp := cfg.RopeParameters; rp != nil {
		if sp := rp["sliding_attention"]; sp != nil && sp.RopeTheta > 0 {
			cfg.SlidingRopeBase = sp.RopeTheta
		}
		if fp := rp["full_attention"]; fp != nil {
			if fp.RopeTheta > 0 {
				cfg.FullRopeBase = fp.RopeTheta
			}
			if fp.PartialRotaryFactor > 0 {
				// Proportional RoPE: the reference computes inv_freq with divisor
				// global_head_dim, then applies rotate_half which splits at head_dim/2.
				// MLX fast_rope splits at dims/2, so we use dims=global_head_dim
				// and pass custom frequencies that match the reference formula.
				// Non-rotated dims use 1e10 so MLX reciprocals to ~0 (identity).
				ghd := int(cfg.GlobalHeadDim)
				cfg.FullRopeDims = ghd
				halfDim := ghd / 2
				ropeAngles := int(fp.PartialRotaryFactor * float32(ghd) / 2)
				freqs := make([]float32, halfDim)
				for i := range ropeAngles {
					freqs[i] = float32(math.Pow(float64(cfg.FullRopeBase), float64(2*i)/float64(ghd)))
				}
				for i := ropeAngles; i < halfDim; i++ {
					freqs[i] = 1e10
				}
				cfg.FullRopeFreqs = mlx.FromValues(freqs, halfDim)
				mlx.Eval(cfg.FullRopeFreqs)
			}
		}
	}

	// Precompute constant scale factors used in forward pass.
	cfg.EmbedScale = float32(math.Sqrt(float64(cfg.HiddenSize)))
	if cfg.HiddenSizePerLayer > 0 {
		cfg.PLEScale = float32(math.Sqrt(float64(cfg.HiddenSizePerLayer)))
		cfg.PLEProjScale = float32(1.0 / math.Sqrt(float64(cfg.HiddenSize)))
		cfg.PLECombineScale = float32(math.Pow(2.0, -0.5))
	}
	cfg.RouterScale = float32(1.0 / math.Sqrt(float64(cfg.HiddenSize)))

	// Compute KV sharing map.
	cfg.KVShareMap = make(map[int32]int32)
	cfg.KVDonors = make(map[int32]bool)
	if cfg.NumKVSharedLayers > 0 && len(cfg.LayerTypes) > 0 {
		firstShared := cfg.NumHiddenLayers - cfg.NumKVSharedLayers
		prevLayers := cfg.LayerTypes[:firstShared]

		for i := firstShared; i < cfg.NumHiddenLayers; i++ {
			layerType := cfg.LayerTypes[i]
			// Find the last non-shared layer of the same type.
			donor := int32(-1)
			for j := len(prevLayers) - 1; j >= 0; j-- {
				if prevLayers[j] == layerType {
					donor = int32(j)
					break
				}
			}
			if donor >= 0 {
				cfg.KVShareMap[i] = donor
				cfg.KVDonors[donor] = true
			}
		}
	}

	return cfg, nil
}

func (m *Model) EnableCompile() bool {
	return true
}

func resolveWeightPrefix(tensors map[string]*mlx.Array) string {
	for _, prefix := range []string{"", "language_model.", "model.language_model."} {
		if tensors[prefix+"embed_tokens.weight"] != nil {
			return prefix
		}
	}
	// Also try with "model." before the layer path.
	for _, prefix := range []string{"", "language_model.", "model.language_model."} {
		if tensors[prefix+"model.embed_tokens.weight"] != nil {
			return prefix + "model."
		}
	}
	return ""
}

func isLayerSliding(layerIdx int32, cfg *TextConfig) bool {
	if len(cfg.LayerTypes) > 0 && int(layerIdx) < len(cfg.LayerTypes) {
		return cfg.LayerTypes[layerIdx] == "sliding_attention"
	}
	if cfg.SlidingWindowPattern <= 0 {
		return false
	}
	return (layerIdx+1)%cfg.SlidingWindowPattern != 0
}

// precomputeGemmaScaledWeights assigns raw norm weights to the *Scaled fields.
// Gemma 4 uses scale_shift=0.0 for all norms (no +1.0 adjustment), so the
// precomputed weights are just the raw weights from the model.
func precomputeGemmaScaledWeights(m *Model) {
	if m.Norm != nil {
		m.NormScaled = m.Norm.Weight
	}

	if m.PerLayerProjNorm != nil {
		m.PerLayerProjNormWeight = m.PerLayerProjNorm.Weight
	}

	for _, layer := range m.Layers {
		if layer == nil || layer.Attention == nil {
			continue
		}

		if layer.InputNorm != nil {
			layer.InputNormScaled = layer.InputNorm.Weight
		}
		if layer.PostAttnNorm != nil {
			layer.PostAttnNormScaled = layer.PostAttnNorm.Weight
		}
		if layer.PreFFNorm != nil {
			layer.PreFFNormScaled = layer.PreFFNorm.Weight
		}
		if layer.PostFFNorm != nil {
			layer.PostFFNormScaled = layer.PostFFNorm.Weight
		}
		if layer.Attention.QNorm != nil {
			layer.Attention.QNormScaled = layer.Attention.QNorm.Weight
		}
		if layer.Attention.KNorm != nil {
			layer.Attention.KNormScaled = layer.Attention.KNorm.Weight
		}
		if layer.PLE != nil && layer.PLE.PostNorm != nil {
			layer.PLE.PostNormScaled = layer.PLE.PostNorm.Weight
		}
		if layer.PostFFNorm1 != nil {
			layer.PostFFNorm1Scaled = layer.PostFFNorm1.Weight
		}
		if layer.PostFFNorm2 != nil {
			layer.PostFFNorm2Scaled = layer.PostFFNorm2.Weight
		}
		if layer.PreFFNorm2 != nil {
			layer.PreFFNorm2Scaled = layer.PreFFNorm2.Weight
		}
	}
}

func newModel(root *model.Root) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	cfg, err := parseTextConfig(configData)
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
		Layers:     make([]*DecoderLayer, cfg.NumHiddenLayers),
		TextConfig: &cfg,
		tok:        tok,
	}

	for i := range m.Layers {
		donor, isShared := cfg.KVShareMap[int32(i)]
		if !isShared {
			donor = -1
		}
		m.Layers[i] = &DecoderLayer{
			LayerIdx:     int32(i),
			IsSliding:    isLayerSliding(int32(i), m.TextConfig),
			KVShareDonor: donor,
			IsDonor:      cfg.KVDonors[int32(i)],
		}
	}

	return m, nil
}

// LoadWeights receives all tensors loaded from the manifest and assigns them
// to model fields.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	m.weightPrefix = resolveWeightPrefix(tensors)
	prefix := m.weightPrefix
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	// Embeddings.
	embedTokens := model.MakeEmbeddingLayer(tensors, prefix+"embed_tokens", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	if embedTokens == nil {
		return fmt.Errorf("missing embedding weight: %sembed_tokens.weight", prefix)
	}
	m.EmbedTokens = embedTokens

	// Final norm.
	normWeight := tensors[prefix+"norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: %snorm.weight", prefix)
	}
	m.Norm = nn.NewRMSNorm(normWeight, m.RMSNormEps)

	// LM head.
	if lmHead := linears.Make(prefix + "lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else if lmHead := linears.Make("lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else {
		// Gemma 4 ties output projection to embeddings.
		m.LMHead = m.EmbedTokens.AsLinear()
	}

	// PLE model-level weights.
	if m.HiddenSizePerLayer > 0 {
		pleEmbed := model.MakeEmbeddingLayer(tensors, prefix+"embed_tokens_per_layer", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
		if pleEmbed == nil {
			return fmt.Errorf("missing PLE embedding weight")
		}
		m.EmbedTokensPerLayer = pleEmbed

		m.PerLayerModelProj = linears.Make(prefix + "per_layer_model_projection")
		if m.PerLayerModelProj == nil {
			return fmt.Errorf("missing per_layer_model_projection weight")
		}

		pleProjNormWeight := tensors[prefix+"per_layer_projection_norm.weight"]
		if pleProjNormWeight == nil {
			return fmt.Errorf("missing per_layer_projection_norm weight")
		}
		m.PerLayerProjNorm = nn.NewRMSNorm(pleProjNormWeight, m.RMSNormEps)
	}

	// Decoder layers.
	for i := range m.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("%slayers.%d", prefix, i)
		isSliding := isLayerSliding(i, m.TextConfig)

		donor, isShared := m.KVShareMap[i]
		if !isShared {
			donor = -1
		}

		layer := &DecoderLayer{
			LayerIdx:     i,
			IsSliding:    isSliding,
			KVShareDonor: donor,
			IsDonor:      m.KVDonors[i],
			Attention:    &Attention{},
			MLP:          &MLP{},
		}

		// Norms.
		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.InputNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.PostAttnNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".pre_feedforward_layernorm.weight"]; w != nil {
			layer.PreFFNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_feedforward_layernorm.weight"]; w != nil {
			layer.PostFFNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}

		// Attention projections.
		layer.Attention.QProj = linears.Make(layerPrefix + ".self_attn.q_proj")
		layer.Attention.KProj = linears.Make(layerPrefix + ".self_attn.k_proj")
		layer.Attention.VProj = linears.Make(layerPrefix + ".self_attn.v_proj")
		layer.Attention.OProj = linears.Make(layerPrefix + ".self_attn.o_proj")

		if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
			layer.Attention.QNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.k_norm.weight"]; w != nil {
			layer.Attention.KNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}

		// MLP.
		layer.MLP.GateProj = linears.Make(layerPrefix + ".mlp.gate_proj")
		layer.MLP.UpProj = linears.Make(layerPrefix + ".mlp.up_proj")
		layer.MLP.DownProj = linears.Make(layerPrefix + ".mlp.down_proj")

		// Layer scalar (all layers in new weights, was full-attention only in earlier releases).
		if w := tensors[layerPrefix+".layer_scalar"]; w != nil {
			layer.LayerScalar = w
		}

		// MoE components.
		if m.EnableMoeBlock {
			// Router.
			routerProj := linears.Make(layerPrefix + ".router.proj")
			// Raw safetensors uses ".router.scale"; runner.go remaps to "_scale".
			routerScale := tensors[layerPrefix+".router.scale"]
			if routerScale == nil {
				routerScale = tensors[layerPrefix+".router_scale"]
			}
			if routerProj == nil || routerScale == nil {
				return fmt.Errorf("layer %d: missing router weights", i)
			}
			layer.Router = &Router{
				Proj:  routerProj,
				Scale: routerScale,
			}

			// MoE expert weights. Try pre-stacked (BF16 from HF) first,
			// then per-expert (from quantized create path).
			perExpertScale := tensors[layerPrefix+".router.per_expert_scale"]
			if perExpertScale == nil {
				perExpertScale = tensors[layerPrefix+".moe.per_expert_scale"]
			}
			if perExpertScale == nil {
				return fmt.Errorf("layer %d: missing MoE per_expert_scale", i)
			}

			moe := &MoEBlock{PerExpertScale: perExpertScale}

			// Check for pre-stacked tensors (unquantized HF format).
			// Try .experts. first (new weight drop), fall back to .moe. (old format).
			gateUpW := tensors[layerPrefix+".experts.gate_up_proj"]
			if gateUpW == nil {
				gateUpW = tensors[layerPrefix+".moe.gate_up_proj"]
			}
			gateW := tensors[layerPrefix+".experts.gate_proj"]
			if gateW == nil {
				gateW = tensors[layerPrefix+".moe.gate_proj"]
			}
			if gateUpW != nil {
				// Fused gate+up: split along dim 1, transpose for GatherMM.
				dims := gateUpW.Dims()
				half := int32(dims[1] / 2)
				gateSlice := sliceAxis1(gateUpW, 0, half)
				upSlice := sliceAxis1(gateUpW, half, int32(dims[1]))
				moe.GateWeight = transposeForGatherMM(gateSlice)
				moe.UpWeight = transposeForGatherMM(upSlice)
				downW := tensors[layerPrefix+".experts.down_proj"]
				if downW == nil {
					downW = tensors[layerPrefix+".moe.down_proj"]
				}
				if downW == nil {
					return fmt.Errorf("layer %d: missing MoE down_proj with fused gate_up_proj", i)
				}
				moe.DownWeight = transposeForGatherMM(downW)
			} else if gateW != nil {
				// Separate gate_proj and up_proj (older format). Transpose for GatherMM.
				moe.GateWeight = transposeForGatherMM(gateW)
				upW := tensors[layerPrefix+".experts.up_proj"]
				if upW == nil {
					upW = tensors[layerPrefix+".moe.up_proj"]
				}
				downW := tensors[layerPrefix+".experts.down_proj"]
				if downW == nil {
					downW = tensors[layerPrefix+".moe.down_proj"]
				}
				moe.UpWeight = transposeForGatherMM(upW)
				moe.DownWeight = transposeForGatherMM(downW)
				if moe.UpWeight == nil || moe.DownWeight == nil {
					return fmt.Errorf("layer %d: incomplete pre-stacked MoE weights", i)
				}
			} else if switchGateUp := firstNonNil(tensors,
				layerPrefix+".moe.switch_mlp.gate_up_proj.weight",
				layerPrefix+".moe.switch_mlp.gate_up_proj"); switchGateUp != nil {
				// Stacked switch_mlp format (from create pipeline with expert packing).
				switchDown := firstNonNil(tensors,
					layerPrefix+".moe.switch_mlp.down_proj.weight",
					layerPrefix+".moe.switch_mlp.down_proj")
				if switchDown == nil {
					return fmt.Errorf("layer %d: missing switch_mlp down_proj", i)
				}

				// Check for quantized weights (scales present).
				// The scale key depends on whether the tensor has .weight suffix.
				gateUpKey := layerPrefix + ".moe.switch_mlp.gate_up_proj.weight"
				if tensors[gateUpKey] == nil {
					gateUpKey = layerPrefix + ".moe.switch_mlp.gate_up_proj"
				}
				downKey := layerPrefix + ".moe.switch_mlp.down_proj.weight"
				if tensors[downKey] == nil {
					downKey = layerPrefix + ".moe.switch_mlp.down_proj"
				}
				gateUpScales := firstNonNil(tensors, gateUpKey+"_scale", gateUpKey+".scale")
				downScales := firstNonNil(tensors, downKey+"_scale", downKey+".scale")

				if gateUpScales != nil && downScales != nil {
					// Quantized: keep fused gate_up as single tensor for GatherQMM.
					// One fused call instead of two separate gate+up calls.
					gateUpBiases := firstNonNil(tensors, gateUpKey+"_qbias", gateUpKey+".bias")
					downBiases := firstNonNil(tensors, downKey+"_qbias", downKey+".bias")

					moe.GateUpWeightQ = switchGateUp
					moe.GateUpScales = gateUpScales
					moe.GateUpBiases = gateUpBiases
					moe.DownWeightQ = switchDown
					moe.DownScales = downScales
					if downBiases != nil {
						moe.DownBiases = downBiases
					}

					groupSize, bits, mode := model.ResolveLinearQuantParams(
						m.QuantGroupSize, m.QuantBits, m.QuantMode,
						m.TensorQuant, gateUpKey, switchGateUp, gateUpScales,
					)
					moe.UseQuantized = true
					moe.UseFusedGateUp = true
					moe.GateUpGroupSize = groupSize
					moe.GateUpBits = bits
					moe.QuantMode = mode

					dGroupSize, dBits, dMode := model.ResolveLinearQuantParams(
						m.QuantGroupSize, m.QuantBits, m.QuantMode,
						m.TensorQuant, downKey, switchDown, downScales,
					)
					moe.DownGroupSize = dGroupSize
					moe.DownBits = dBits
					moe.DownQuantMode = dMode
				} else {
					// Unquantized switch_mlp: keep fused and transpose for GatherMM.
					moe.GateUpWeight = transposeForGatherMM(switchGateUp)
					moe.UseFusedGateUp = true
					moe.DownWeight = transposeForGatherMM(switchDown)
				}
			} else {
				// Per-expert tensors (from create path).
				// Try separate gate_proj/up_proj first, then fused gate_up_proj.
				gateStacked := collectExpertProjection(tensors, m.TextConfig,
					layerPrefix+".moe.experts", "gate_proj", m.NumExperts)
				upStacked := collectExpertProjection(tensors, m.TextConfig,
					layerPrefix+".moe.experts", "up_proj", m.NumExperts)
				downStacked := collectExpertProjection(tensors, m.TextConfig,
					layerPrefix+".moe.experts", "down_proj", m.NumExperts)

				if gateStacked == nil && upStacked == nil {
					// Try fused gate_up_proj format — split along axis 1 (out-dim).
					// For quantized weights, also split scales and biases.
					gateUpStacked := collectExpertProjection(tensors, m.TextConfig,
						layerPrefix+".moe.experts", "gate_up_proj", m.NumExperts)
					if gateUpStacked != nil {
						dims := gateUpStacked.Weight.Dims()
						if len(dims) >= 2 {
							mid := int32(dims[1] / 2)
							gateStacked = &stackedExpertResult{
								Weight:    sliceAxis1(gateUpStacked.Weight, 0, mid),
								Bits:      gateUpStacked.Bits,
								GroupSize: gateUpStacked.GroupSize,
								Mode:      gateUpStacked.Mode,
							}
							upStacked = &stackedExpertResult{
								Weight:    sliceAxis1(gateUpStacked.Weight, mid, int32(dims[1])),
								Bits:      gateUpStacked.Bits,
								GroupSize: gateUpStacked.GroupSize,
								Mode:      gateUpStacked.Mode,
							}
							if gateUpStacked.Scales != nil {
								sDims := gateUpStacked.Scales.Dims()
								sMid := int32(sDims[1] / 2)
								gateStacked.Scales = sliceAxis1(gateUpStacked.Scales, 0, sMid)
								upStacked.Scales = sliceAxis1(gateUpStacked.Scales, sMid, int32(sDims[1]))
							}
							if gateUpStacked.Biases != nil {
								bDims := gateUpStacked.Biases.Dims()
								bMid := int32(bDims[1] / 2)
								gateStacked.Biases = sliceAxis1(gateUpStacked.Biases, 0, bMid)
								upStacked.Biases = sliceAxis1(gateUpStacked.Biases, bMid, int32(bDims[1]))
							}
						}
					}
				}

				if gateStacked == nil || upStacked == nil || downStacked == nil {
					return fmt.Errorf("layer %d: missing MoE expert weights", i)
				}
				// Use GatherQMM if all projections have quantized weights.
				if gateStacked.Scales != nil && upStacked.Scales != nil && downStacked.Scales != nil {
					moe.GateWeightQ = gateStacked.Weight
					moe.GateScales = gateStacked.Scales
					moe.GateBiases = gateStacked.Biases
					moe.UpWeightQ = upStacked.Weight
					moe.UpScales = upStacked.Scales
					moe.UpBiases = upStacked.Biases
					moe.DownWeightQ = downStacked.Weight
					moe.DownScales = downStacked.Scales
					moe.DownBiases = downStacked.Biases
					moe.UseQuantized = true
					moe.GateGroupSize = gateStacked.GroupSize
					moe.GateBits = gateStacked.Bits
					moe.UpGroupSize = upStacked.GroupSize
					moe.UpBits = upStacked.Bits
					moe.DownGroupSize = downStacked.GroupSize
					moe.DownBits = downStacked.Bits
					moe.QuantMode = gateStacked.Mode
					moe.DownQuantMode = downStacked.Mode
				} else {
					// Unquantized: transpose for GatherMM (expects [experts, in, out]).
					moe.GateWeight = transposeForGatherMM(gateStacked.Weight)
					moe.UpWeight = transposeForGatherMM(upStacked.Weight)
					moe.DownWeight = transposeForGatherMM(downStacked.Weight)
				}
			}
			layer.MoE = moe

			// Additional norms for MoE dual-path.
			if w := tensors[layerPrefix+".post_feedforward_layernorm_1.weight"]; w != nil {
				layer.PostFFNorm1 = nn.NewRMSNorm(w, m.RMSNormEps)
			}
			if w := tensors[layerPrefix+".post_feedforward_layernorm_2.weight"]; w != nil {
				layer.PostFFNorm2 = nn.NewRMSNorm(w, m.RMSNormEps)
			}
			if w := tensors[layerPrefix+".pre_feedforward_layernorm_2.weight"]; w != nil {
				layer.PreFFNorm2 = nn.NewRMSNorm(w, m.RMSNormEps)
			}

			if layer.PostFFNorm1 == nil || layer.PostFFNorm2 == nil || layer.PreFFNorm2 == nil {
				return fmt.Errorf("layer %d: missing MoE norm weights", i)
			}
		}

		// PLE per-layer weights.
		if m.HiddenSizePerLayer > 0 {
			layer.PLE = &PLELayer{}
			layer.PLE.InputGate = linears.Make(layerPrefix + ".per_layer_input_gate")
			layer.PLE.Projection = linears.Make(layerPrefix + ".per_layer_projection")
			if w := tensors[layerPrefix+".post_per_layer_input_norm.weight"]; w != nil {
				layer.PLE.PostNorm = nn.NewRMSNorm(w, m.RMSNormEps)
			}

			if layer.PLE.InputGate == nil || layer.PLE.Projection == nil || layer.PLE.PostNorm == nil {
				return fmt.Errorf("layer %d: missing PLE weights", i)
			}
		}

		// Validation.
		if layer.InputNorm == nil {
			return fmt.Errorf("layer %d: missing input_layernorm", i)
		}
		if layer.PostAttnNorm == nil {
			return fmt.Errorf("layer %d: missing post_attention_layernorm", i)
		}
		if layer.PreFFNorm == nil {
			return fmt.Errorf("layer %d: missing pre_feedforward_layernorm", i)
		}
		if layer.PostFFNorm == nil {
			return fmt.Errorf("layer %d: missing post_feedforward_layernorm", i)
		}
		if layer.Attention.QProj == nil || layer.Attention.OProj == nil {
			return fmt.Errorf("layer %d: missing attention q/o projections", i)
		}
		if layer.Attention.KProj == nil {
			return fmt.Errorf("layer %d: missing attention k projection", i)
		}
		// VProj is nil for K=V full-attention layers (value_states = key_states).
		useAltAttn := m.AttentionKEqV && !isSliding
		if layer.Attention.VProj == nil && !useAltAttn {
			return fmt.Errorf("layer %d: missing attention v projection", i)
		}
		if layer.Attention.QNorm == nil || layer.Attention.KNorm == nil {
			return fmt.Errorf("layer %d: missing attention q/k norms", i)
		}
		if layer.MLP.GateProj == nil || layer.MLP.UpProj == nil || layer.MLP.DownProj == nil {
			return fmt.Errorf("layer %d: missing mlp projections", i)
		}

		m.Layers[i] = layer
	}

	precomputeGemmaScaledWeights(m)
	if m.NormScaled == nil {
		return fmt.Errorf("missing precomputed final norm weight")
	}

	return nil
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	h := m.EmbedTokens.Forward(b.InputIDs)
	h = mlx.MulScalar(h, m.EmbedScale)

	// Compute PLE inputs if configured.
	var perLayerInputs *mlx.Array
	if m.HiddenSizePerLayer > 0 && m.EmbedTokensPerLayer != nil {
		perLayerInputs = m.computePLEInputs(b.InputIDs, h)
	}

	// KV sharing: each donor layer stores its KVHistory here so later
	// shared layers can reuse it in lieu of their own cache update.
	var sharedKV map[int32]sharedHistory
	if len(m.KVShareMap) > 0 {
		sharedKV = make(map[int32]sharedHistory)
	}

	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}

		// Extract per-layer PLE input for this layer.
		var pleInput *mlx.Array
		if perLayerInputs != nil {
			pleInput = sliceLayerDim(perLayerInputs, int32(i), B, L, m.HiddenSizePerLayer)
		}

		// Get shared KV for this layer if it's a shared layer.
		var donor *sharedHistory
		if layer.KVShareDonor >= 0 {
			if d, ok := sharedKV[layer.KVShareDonor]; ok {
				donor = &d
			}
		}

		var donorKV *sharedHistory
		h, donorKV = layer.Forward(h, b, c, positions, B, L, m.TextConfig, pleInput, donor)

		// If this layer is a donor, store its cached KV for later shared layers.
		if layer.IsDonor && donorKV != nil {
			sharedKV[layer.LayerIdx] = *donorKV
		}
	}

	return mlx.RMSNormFn(h, m.NormScaled, m.RMSNormEps)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	logits := m.LMHead.Forward(x)

	if m.FinalLogitSoftcapping > 0 {
		cap := mlx.FromValue(m.FinalLogitSoftcapping).AsType(logits.DType())
		logits = mlx.LogitSoftcap(logits, cap)
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

// NewCaches creates cache objects for layers that own KV state.
func (m *Model) NewCaches() []cache.Cache {
	cacheLayers := len(m.Layers)
	for i, layer := range m.Layers {
		if layer.KVShareDonor >= 0 {
			cacheLayers = i
			break
		}
	}

	caches := make([]cache.Cache, cacheLayers)
	for i, layer := range m.Layers[:cacheLayers] {
		if m.SlidingWindow > 0 && layer.IsSliding {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

// computePLEInputs computes per-layer embeddings and projections.
// Returns shape [B, L, NumHiddenLayers, HiddenSizePerLayer].
func (m *Model) computePLEInputs(tokens, h *mlx.Array) *mlx.Array {
	dims := tokens.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	pleScale := m.PLEScale
	projScale := m.PLEProjScale

	// Token-based per-layer embeddings: [B, L, NumLayers*PLEDim]
	pleEmb := m.EmbedTokensPerLayer.Forward(tokens)
	pleEmb = mlx.MulScalar(pleEmb, pleScale)
	// Reshape to [B, L, NumLayers, PLEDim]
	pleEmb = mlx.Reshape(pleEmb, B, L, m.NumHiddenLayers, m.HiddenSizePerLayer)

	// Hidden-state projection: [B, L, NumLayers*PLEDim]
	pleProj := m.PerLayerModelProj.Forward(h)
	pleProj = mlx.MulScalar(pleProj, projScale)
	// Reshape to [B, L, NumLayers, PLEDim]
	pleProj = mlx.Reshape(pleProj, B, L, m.NumHiddenLayers, m.HiddenSizePerLayer)

	// Apply per-layer projection norm (scale_shift=0.0, uses raw weight).
	pleProj = mlx.RMSNormFn(pleProj, m.PerLayerProjNormWeight, m.RMSNormEps)

	// Combine: (proj + emb) * 2^(-0.5)
	combined := mlx.Add(pleProj, pleEmb)
	combined = mlx.MulScalar(combined, m.PLECombineScale)

	return combined
}

// sliceLayerDim extracts a single layer's PLE input from the combined tensor.
// Input shape: [B, L, NumLayers, PLEDim], output shape: [B, L, PLEDim].
func sliceLayerDim(combined *mlx.Array, layerIdx, B, L, pleDim int32) *mlx.Array {
	sliced := mlx.SliceStartStop(combined,
		[]int32{0, 0, layerIdx, 0},
		[]int32{B, L, layerIdx + 1, pleDim},
	)
	return mlx.Squeeze(sliced, 2)
}

func (l *DecoderLayer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *TextConfig, pleInput *mlx.Array, donor *sharedHistory) (*mlx.Array, *sharedHistory) {
	normed := mlx.RMSNormFn(x, l.InputNormScaled, cfg.RMSNormEps)
	attnOut, kv := l.Attention.Forward(normed, b, c, positions, B, L, l.IsSliding, cfg, donor)
	attnOut = mlx.RMSNormFn(attnOut, l.PostAttnNormScaled, cfg.RMSNormEps)
	h := mlx.Add(x, attnOut)

	if l.Router != nil && l.MoE != nil {
		// Dual-path: dense MLP + MoE, both normed separately, then combined.
		residual := h

		// Path 1: Dense MLP.
		normed = mlx.RMSNormFn(h, l.PreFFNormScaled, cfg.RMSNormEps)
		mlpOut := l.MLP.Forward(normed)
		mlpOut = mlx.RMSNormFn(mlpOut, l.PostFFNorm1Scaled, cfg.RMSNormEps)

		// Path 2: MoE.
		scores, inds := l.Router.Forward(h, cfg)
		normed2 := mlx.RMSNormFn(h, l.PreFFNorm2Scaled, cfg.RMSNormEps)
		moeOut := l.MoE.Forward(normed2, scores, inds, cfg)
		moeOut = mlx.RMSNormFn(moeOut, l.PostFFNorm2Scaled, cfg.RMSNormEps)

		// Combine and apply outer post-norm.
		combined := mlx.Add(mlpOut, moeOut)
		combined = mlx.RMSNormFn(combined, l.PostFFNormScaled, cfg.RMSNormEps)
		h = mlx.Add(residual, combined)
	} else {
		// Standard single MLP path.
		normed = mlx.RMSNormFn(h, l.PreFFNormScaled, cfg.RMSNormEps)
		mlpOut := l.MLP.Forward(normed)
		mlpOut = mlx.RMSNormFn(mlpOut, l.PostFFNormScaled, cfg.RMSNormEps)
		h = mlx.Add(h, mlpOut)
	}

	// PLE injection (after MLP residual).
	if l.PLE != nil && pleInput != nil {
		residual := h
		gated := mlx.GeGLU(l.PLE.InputGate.Forward(h), pleInput)
		projected := l.PLE.Projection.Forward(gated)
		projected = mlx.RMSNormFn(projected, l.PLE.PostNormScaled, cfg.RMSNormEps)
		h = mlx.Add(residual, projected)
	}

	// Layer scalar for full-attention layers.
	if l.LayerScalar != nil {
		h = mlx.Mul(h, l.LayerScalar)
	}

	return h, kv
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, isSliding bool, cfg *TextConfig, donor *sharedHistory) (*mlx.Array, *sharedHistory) {
	// Determine head dim and scale based on layer type.
	headDim := cfg.HeadDim
	scale := cfg.SlidingScale
	ropeDims := cfg.SlidingRopeDims
	ropeBase := cfg.SlidingRopeBase
	if !isSliding {
		headDim = cfg.GlobalHeadDim
		scale = cfg.FullScale
		ropeDims = cfg.FullRopeDims
		ropeBase = cfg.FullRopeBase
	}

	q := a.QProj.Forward(x)
	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, headDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)

	// Apply Q norm.
	q = mlx.RMSNormFn(q, a.QNormScaled, cfg.RMSNormEps)

	var ropeFreqs *mlx.Array
	if !isSliding {
		ropeFreqs = cfg.FullRopeFreqs
	}
	q = mlx.RoPEWithFreqs(q, ropeDims, false, ropeBase, 1.0, positions, ropeFreqs)

	kv := donor
	if kv == nil {
		// Determine KV head count: K=V full-attention layers use NumGlobalKeyValueHeads.
		kvHeads := cfg.NumKeyValueHeads
		if a.VProj == nil && !isSliding && cfg.NumGlobalKeyValueHeads > 0 {
			kvHeads = cfg.NumGlobalKeyValueHeads
		}

		k := a.KProj.Forward(x)
		k = mlx.Reshape(k, B, L, kvHeads, headDim)
		k = mlx.Transpose(k, 0, 2, 1, 3)

		var v *mlx.Array
		if a.VProj != nil {
			v = a.VProj.Forward(x)
			v = mlx.Reshape(v, B, L, kvHeads, headDim)
			v = mlx.Transpose(v, 0, 2, 1, 3)
		} else {
			// K=V: value_states = key_states (raw, before k_norm/rope).
			v = k
		}

		// Apply K norm.
		k = mlx.RMSNormFn(k, a.KNormScaled, cfg.RMSNormEps)

		// Apply RoPE to K.
		k = mlx.RoPEWithFreqs(k, ropeDims, false, ropeBase, 1.0, positions, ropeFreqs)

		// Apply V norm (no learnable weight, pure RMS normalization).
		v = mlx.RMSNormFn(v, nil, cfg.RMSNormEps)

		// Update cache.
		kv = &sharedHistory{}
		if c != nil {
			kv.history = c.(cache.Attention).Update(b, k, v)
		} else {
			kv.k, kv.v = k, v
			if isSliding {
				kv.mask = nn.SlidingWindowMask(b, k.Dim(2), int(cfg.SlidingWindow), q.DType())
			}
		}
	}

	var out *mlx.Array
	if headDim > 128 && L > 1 && !mlx.MetalIsAvailable() {
		// Manual attention for CUDA prefill with head_dim > 128.
		// cuDNN SDPA requires head_dim <= 128, and the MLX CUDA SDPA vector
		// kernel only handles L < 4 (generation). For prefill, we fall back
		// to explicit matmul+softmax+matmul on CUDA.
		var k, v *mlx.Array
		mask := nn.CausalMask().Intersect(nn.QPaddingMask(b, q.DType()))
		if kv.history != nil {
			k, v = kv.history.K(), kv.history.V()
			mask = kv.history.Mask(mask)
		} else {
			k, v = kv.k, kv.v
			mask = mask.Intersect(nn.KPaddingMask(b, k.Dim(2), b.SeqQueryLens, q.DType()))
			mask = mask.Intersect(kv.mask)
		}
		kvHeads := int32(k.Dim(1))
		nRepeats := cfg.NumAttentionHeads / kvHeads
		kLen := int32(k.Dim(2))
		// AsArray returns [B, 1, L, K]; reshape to rank 5 so that
		// right-to-left broadcast against scores [B, kvHeads,
		// nRepeats, L, K] aligns the batch dim correctly.
		maskArr := mlx.Reshape(mask.AsArray(b, int(kLen), q.DType()), B, 1, 1, L, kLen)

		q = mlx.MulScalar(q, scale)
		q = mlx.Reshape(q, B, kvHeads, nRepeats, L, headDim)
		k = mlx.Reshape(k, B, kvHeads, 1, kLen, headDim)
		v = mlx.Reshape(v, B, kvHeads, 1, kLen, headDim)

		kT := mlx.Transpose(k, 0, 1, 2, 4, 3)
		scores := mlx.Matmul(q, kT)
		scores = mlx.Add(scores, maskArr)
		scores = mlx.SoftmaxAxis(scores, -1, true)
		out = mlx.Matmul(scores, v)
		out = mlx.Reshape(out, B, cfg.NumAttentionHeads, L, headDim)
	} else {
		var opt nn.SDPAOption
		mask := nn.CausalMask()
		if kv.history != nil {
			opt = nn.WithKVHistory(kv.history)
		} else {
			opt = nn.WithKV(kv.k, kv.v, b.SeqQueryLens)
			mask = mask.Intersect(kv.mask)
		}
		out = nn.ScaledDotProductAttention(b, q, scale, opt, nn.WithMask(mask))
	}
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*headDim)
	if !mlx.MetalIsAvailable() {
		// Force contiguous layout before OProj on CUDA where matmul handles
		// strided views differently. Metal handles them natively.
		out = mlx.Contiguous(out, false)
	}
	return a.OProj.Forward(out), kv
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	gate := m.GateProj.Forward(x)
	up := m.UpProj.Forward(x)
	return m.DownProj.Forward(mlx.GeGLU(gate, up))
}

// Forward runs the router to select top-k experts per token.
// Returns (scores [B*L, topK], indices [B*L, topK]).
func (r *Router) Forward(x *mlx.Array, cfg *TextConfig) (*mlx.Array, *mlx.Array) {
	dims := x.Dims()
	BL := int32(dims[0]) * int32(dims[1])

	// Flatten to [B*L, hidden].
	xFlat := mlx.Reshape(x, BL, cfg.HiddenSize)

	// Norm (no weight) -> scale by 1/sqrt(hidden_size) -> multiply by learnable scale.
	normed := mlx.RMSNormFn(xFlat, nil, cfg.RMSNormEps)
	normed = mlx.MulScalar(normed, cfg.RouterScale)
	normed = mlx.Mul(normed, r.Scale)

	// Project to expert scores: [B*L, num_experts].
	expertScores := r.Proj.Forward(normed)

	// Top-k selection via argpartition on negated scores.
	neg := mlx.Neg(expertScores)
	inds := mlx.Argpartition(neg, int(cfg.TopKExperts)-1, -1)
	inds = mlx.SliceStartStop(inds,
		[]int32{0, 0},
		[]int32{BL, cfg.TopKExperts},
	)

	// Softmax only over selected logits. This is equivalent to full softmax +
	// gather + renormalize, but avoids normalizing over every expert.
	scores := mlx.TakeAlongAxis(expertScores, inds, -1)
	scores = mlx.SoftmaxAxis(scores, -1, true) // [B*L, topK]

	return scores, inds
}

// Forward runs the MoE block using GatherQMM (quantized) or GatherMM (dense).
// scores: [B*L, topK], inds: [B*L, topK], x: [B, L, hidden].
func (m *MoEBlock) Forward(x *mlx.Array, scores, inds *mlx.Array, cfg *TextConfig) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := cfg.TopKExperts

	// Flatten and prepare for expert dispatch.
	xFlat := mlx.Reshape(x, B*L, 1, 1, cfg.HiddenSize)
	idxFlat := mlx.Reshape(inds, B*L, topK)

	// Sort indices for efficiency when batch is large enough.
	// The sorted_indices flag tells GatherQMM the indices are pre-sorted,
	// enabling coalesced memory access. Testing confirmed the sort is
	// beneficial for prefill (2x faster with sort at 2048 tokens).
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

	// Expert computation: gate+up followed by GELU and down.
	// When gate+up are fused, we do 2 GatherQMM calls instead of 3.
	var hidden, down *mlx.Array
	if m.UseQuantized {
		if m.UseFusedGateUp {
			// Fused gate+up: single GatherQMM produces [B*L*topK, 1, 1, 2*intermediate]
			gateUp := mlx.GatherQMM(xFlat, m.GateUpWeightQ, m.GateUpScales, m.GateUpBiases,
				nil, idxFlat, true, m.GateUpGroupSize, m.GateUpBits, m.QuantMode, doSort)
			// Split along last dim into gate and up
			guDims := gateUp.Dims()
			mid := int32(guDims[len(guDims)-1] / 2)
			gate := mlx.SliceStartStop(gateUp,
				[]int32{0, 0, 0, 0},
				[]int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), mid})
			up := mlx.SliceStartStop(gateUp,
				[]int32{0, 0, 0, mid},
				[]int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), int32(guDims[len(guDims)-1])})
			hidden = mlx.GeGLU(gate, up)
		} else {
			gate := mlx.GatherQMM(xFlat, m.GateWeightQ, m.GateScales, m.GateBiases,
				nil, idxFlat, true, m.GateGroupSize, m.GateBits, m.QuantMode, doSort)
			up := mlx.GatherQMM(xFlat, m.UpWeightQ, m.UpScales, m.UpBiases,
				nil, idxFlat, true, m.UpGroupSize, m.UpBits, m.QuantMode, doSort)
			hidden = mlx.GeGLU(gate, up)
		}
		downMode := m.DownQuantMode
		if downMode == "" {
			downMode = m.QuantMode
		}
		down = mlx.GatherQMM(hidden, m.DownWeightQ, m.DownScales, m.DownBiases,
			nil, idxFlat, true, m.DownGroupSize, m.DownBits, downMode, doSort)
	} else {
		if m.UseFusedGateUp && m.GateUpWeight != nil {
			gateUp := mlx.GatherMM(xFlat, m.GateUpWeight, nil, idxFlat, doSort)
			guDims := gateUp.Dims()
			mid := int32(guDims[len(guDims)-1] / 2)
			gate := mlx.SliceStartStop(gateUp,
				[]int32{0, 0, 0, 0},
				[]int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), mid})
			up := mlx.SliceStartStop(gateUp,
				[]int32{0, 0, 0, mid},
				[]int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), int32(guDims[len(guDims)-1])})
			hidden = mlx.GeGLU(gate, up)
		} else {
			gate := mlx.GatherMM(xFlat, m.GateWeight, nil, idxFlat, doSort)
			up := mlx.GatherMM(xFlat, m.UpWeight, nil, idxFlat, doSort)
			hidden = mlx.GeGLU(gate, up)
		}
		down = mlx.GatherMM(hidden, m.DownWeight, nil, idxFlat, doSort)
	}

	// Unsort if needed.
	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	// Reshape to [B*L, topK, hidden_size].
	down = mlx.Reshape(down, B*L, topK, cfg.HiddenSize)

	// Gather per-expert scales at selected indices: flatten inds, take, reshape back.
	indsFlat := mlx.Reshape(inds, B*L*topK)
	expertScales := mlx.Take(m.PerExpertScale, indsFlat, 0) // [B*L*topK]
	expertScales = mlx.Reshape(expertScales, B*L, topK)     // [B*L, topK]
	down = mlx.Mul(down, mlx.ExpandDims(expertScales, -1))

	// Weight by dispatch scores and sum across experts (axis 1 = topK dim).
	y := mlx.Sum(mlx.Mul(down, mlx.ExpandDims(scores, -1)), 1, false) // [B*L, hidden_size]

	return mlx.Reshape(y, B, L, cfg.HiddenSize)
}
