// Package granitemoe provides a GraniteMoe-style sparse mixture-of-experts
// decoder-only transformer for MLX.
package granitemoe

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() {
	model.Register("GraniteMoeForCausalLM", newModel)
}

// Config holds GraniteMoe model configuration.
type Config struct {
	HiddenSize            int32              `json:"hidden_size"`
	NumHiddenLayers       int32              `json:"num_hidden_layers"`
	IntermediateSize      int32              `json:"intermediate_size"`
	NumAttentionHeads     int32              `json:"num_attention_heads"`
	NumKeyValueHeads      int32              `json:"num_key_value_heads"`
	VocabSize             int32              `json:"vocab_size"`
	RMSNormEps            float32            `json:"rms_norm_eps"`
	RopeTheta             float32            `json:"rope_theta"`
	MaxPositionEmbeddings int32              `json:"max_position_embeddings"`
	TieWordEmbeddings     bool               `json:"tie_word_embeddings"`

	// Granite-specific multipliers (all default to 1.0, except
	// AttentionMultiplier which defaults to the standard 1/sqrt(head_dim)
	// scale, if unset).
	EmbeddingMultiplier float32 `json:"embedding_multiplier"`
	AttentionMultiplier float32 `json:"attention_multiplier"`
	ResidualMultiplier  float32 `json:"residual_multiplier"`
	LogitsScaling       float32 `json:"logits_scaling"`

	// MoE routing.
	NumLocalExperts  int32 `json:"num_local_experts"`
	NumExpertsPerTok int32 `json:"num_experts_per_tok"`

	// RoPE dual-key: prefer rope_parameters, fall back to rope_scaling.
	RopeParameters *nn.RopeParameters `json:"rope_parameters"`
	RopeScaling    *nn.RopeParameters `json:"rope_scaling"`

	// Quantization parameters (set during load based on model quantization).
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	HeadDim    int32        `json:"-"`
	Scale      float32      `json:"-"`
	RopeFreqs  *mlx.Array   `json:"-"` // nil when no YaRN
	RopeMScale float32      `json:"-"` // 1 when no YaRN
}

// Model is a GraniteMoe text model.
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config

	weightPrefix string
}

type Layer struct {
	Attention     *Attention
	MoE           *SparseMoE
	AttentionNorm *nn.RMSNorm
	MLPNorm       *nn.RMSNorm
}

type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
}

// SparseMoE routes each token to the top-k of NumLocalExperts expert MLPs.
type SparseMoE struct {
	Router    nn.LinearLayer
	SwitchMLP *SwitchMLP
}

// SwitchMLP executes the selected expert MLPs with stacked expert weights.
// GateUpWeight holds the fused [gate; up] projection (matching the
// checkpoint's single input_linear tensor); its output is split in half
// along the last axis before the SwiGLU activation.
type SwitchMLP struct {
	GateUpWeight *mlx.Array
	DownWeight   *mlx.Array

	GateUpWeightQ, GateUpScales, GateUpBiases *mlx.Array
	DownWeightQ, DownScales, DownBiases       *mlx.Array
	GateUpGlobalScales, DownGlobalScales      *mlx.Array

	GateUpBits, DownBits           int
	GateUpGroupSize, DownGroupSize int
	GateUpMode, DownMode           string
}

type stackedExpertWeights struct {
	Weight       *mlx.Array
	Scales       *mlx.Array
	Biases       *mlx.Array
	GlobalScales *mlx.Array
	Bits         int
	GroupSize    int
	Mode         string
}

func resolveWeightPrefix(tensors map[string]*mlx.Array) string {
	for _, prefix := range []string{"", "language_model."} {
		if tensors[prefix+"model.embed_tokens.weight"] != nil {
			return prefix
		}
	}
	return ""
}

// tensorAny returns the first non-nil array from the given keys, and the key
// that matched (for suffix lookups like key+"_scale").
func tensorAny(tensors map[string]*mlx.Array, keys ...string) (*mlx.Array, string) {
	for _, k := range keys {
		if v := tensors[k]; v != nil {
			return v, k
		}
	}
	return nil, ""
}

// tensorByBase returns the tensor for base+".weight" or base (whichever is
// present), plus the matched key for suffix lookups.
func tensorByBase(tensors map[string]*mlx.Array, base string) (*mlx.Array, string) {
	return tensorAny(tensors, base+".weight", base)
}

func newModel(root *model.Root) (model.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if cfg.HiddenSize <= 0 {
		return nil, fmt.Errorf("invalid hidden_size: %d", cfg.HiddenSize)
	}
	if cfg.NumAttentionHeads <= 0 {
		return nil, fmt.Errorf("invalid num_attention_heads: %d", cfg.NumAttentionHeads)
	}
	if cfg.NumKeyValueHeads <= 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return nil, fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("invalid head_dim: %d", cfg.HeadDim)
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return nil, fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-5
	}
	if cfg.EmbeddingMultiplier == 0 {
		cfg.EmbeddingMultiplier = 1.0
	}
	if cfg.ResidualMultiplier == 0 {
		cfg.ResidualMultiplier = 1.0
	}
	if cfg.LogitsScaling == 0 {
		cfg.LogitsScaling = 1.0
	}
	if cfg.AttentionMultiplier == 0 {
		cfg.AttentionMultiplier = 1.0
	}
	// Granite's attention_multiplier replaces the standard 1/sqrt(head_dim)
	// scaling factor used in attention.
	cfg.Scale = cfg.AttentionMultiplier

	if cfg.NumLocalExperts <= 0 {
		cfg.NumLocalExperts = 8
	}
	if cfg.NumExpertsPerTok <= 0 {
		cfg.NumExpertsPerTok = 2
	}
	if cfg.NumExpertsPerTok > cfg.NumLocalExperts {
		return nil, fmt.Errorf("num_experts_per_tok (%d) exceeds num_local_experts (%d)", cfg.NumExpertsPerTok, cfg.NumLocalExperts)
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

	// RoPE dual-key handling: prefer rope_parameters, fall back to rope_scaling.
	ropeParams := cfg.RopeParameters
	if ropeParams == nil {
		ropeParams = cfg.RopeScaling
	}
	cfg.RopeFreqs = nil
	cfg.RopeMScale = 1.0
	if ropeParams != nil && strings.EqualFold(ropeParams.TypeName(), "yarn") {
		cfg.RopeFreqs, cfg.RopeMScale = nn.BuildYarnRopeFreqs(int(cfg.HeadDim), cfg.RopeTheta, ropeParams)
	}

	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer config: %w", err)
	}

	tokConfig := &tokenizer.TokenizerConfig{
		ConfigJSON: configData,
	}
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
	if w == nil || w.NumDims() != 3 {
		return w
	}
	return mlx.Transpose(w, 0, 2, 1).Clone()
}

// fuseExpertStacks concatenates two expert stacks along the given axis.
func fuseExpertStacks(a, b *mlx.Array, axis int) *mlx.Array {
	if a == nil || b == nil {
		return nil
	}
	return mlx.Concatenate([]*mlx.Array{a, b}, axis).Clone()
}

// fuseGateUpProjections joins gate and up stacks along the output dimension,
// which is exact for quantized stacks: groups run along the input dimension,
// and a global scale covers the whole bank row.
func fuseGateUpProjections(gate, up *stackedExpertWeights) *stackedExpertWeights {
	if gate == nil || up == nil {
		return nil
	}
	if gate.Weight == nil || gate.Weight.NumDims() != 3 || up.Weight == nil || up.Weight.NumDims() != 3 {
		return nil
	}
	// Quantized path: only fuse when scales match and metadata aligns.
	// Compare GlobalScales (per-expert scalar bank), not Scales (per-group tensors).
	if gate.Scales != nil && up.Scales != nil &&
		model.SameGlobalScales(gate.GlobalScales, up.GlobalScales) &&
		gate.Bits == up.Bits && gate.GroupSize == up.GroupSize && gate.Mode == up.Mode &&
		(gate.Biases == nil) == (up.Biases == nil) {
		fused := &stackedExpertWeights{
			Weight:       fuseExpertStacks(gate.Weight, up.Weight, 1),
			Scales:       fuseExpertStacks(gate.Scales, up.Scales, 1),
			GlobalScales: gate.GlobalScales,
			Bits:         gate.Bits,
			GroupSize:    gate.GroupSize,
			Mode:         gate.Mode,
		}
		if gate.Biases != nil {
			fused.Biases = fuseExpertStacks(gate.Biases, up.Biases, 1)
		}
		return fused
	}
	// If only one side has scales, fusion is not safe.
	if gate.Scales != nil || up.Scales != nil {
		return nil
	}
	return &stackedExpertWeights{
		Weight:    mlx.Concatenate([]*mlx.Array{gate.Weight, up.Weight}, 1),
		Bits:      gate.Bits,
		GroupSize: gate.GroupSize,
		Mode:      gate.Mode,
	}
}

// loadStackedProjection loads an already-stacked (single tensor covering all
// experts) projection by its base name(s). It tries each base in order
// (first match wins), handling quantized vs plain branches.
func loadStackedProjection(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, bases ...string) *stackedExpertWeights {
	for _, base := range bases {
		w, key := tensorByBase(tensors, base)
		if w == nil {
			continue
		}

		scales := tensors[key+"_scale"]
		if scales == nil {
			return &stackedExpertWeights{Weight: w}
		}

		qbiases := tensors[key+"_qbias"]
		globalScale, _ := model.ReadGlobalScale(tensors, key, base+".weight", base)
		groupSize, bits, mode := model.ResolveLinearQuantParams(
			cfg.QuantGroupSize,
			cfg.QuantBits,
			cfg.QuantMode,
			cfg.TensorQuant,
			key,
			w,
			scales,
		)
		globalScale = model.PrepareGatherQMMGlobalScale(globalScale, w.Dim(0))
		if useQuantized && supportsGatherQMM(mode, bits) {
			return &stackedExpertWeights{
				Weight:       w,
				Scales:       scales,
				Biases:       qbiases,
				GlobalScales: globalScale,
				Bits:         bits,
				GroupSize:    groupSize,
				Mode:         mode,
			}
		}

		if useQuantized {
			slog.Warn("dequantizing expert weights: no gather kernel for format", "tensor", key, "mode", mode, "bits", bits)
		}
		return &stackedExpertWeights{
			Weight:       mlx.Dequantize(w, scales, qbiases, groupSize, bits, mode, globalScale),
			Bits:         bits,
			GroupSize:    groupSize,
			Mode:         mode,
		}
	}

	return nil
}

// combinedGateUpProjection loads the fused block_sparse_moe.input_linear stack;
// nil when the checkpoint ships gate/up separately.
func combinedGateUpProjection(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, layerPrefix string) *stackedExpertWeights {
	gateUp, key := tensorAny(
		tensors,
		layerPrefix+".block_sparse_moe.input_linear.weight",
		layerPrefix+".block_sparse_moe.input_linear",
	)
	if gateUp == nil || gateUp.NumDims() != 3 {
		return nil
	}

	scales := tensors[key+"_scale"]
	if scales == nil {
		return &stackedExpertWeights{Weight: gateUp}
	}

	qbiases := tensors[key+"_qbias"]
	globalScale, _ := model.ReadGlobalScale(tensors, key,
		layerPrefix+".block_sparse_moe.input_linear.weight",
		layerPrefix+".block_sparse_moe.input_linear")
	groupSize, bits, mode := model.ResolveLinearQuantParams(
		cfg.QuantGroupSize,
		cfg.QuantBits,
		cfg.QuantMode,
		cfg.TensorQuant,
		key,
		gateUp,
		scales,
	)
	globalScale = model.PrepareGatherQMMGlobalScale(globalScale, gateUp.Dim(0))
	if useQuantized && supportsGatherQMM(mode, bits) {
		return &stackedExpertWeights{
			Weight:       gateUp,
			Scales:       scales,
			Biases:       qbiases,
			GlobalScales: globalScale,
			Bits:         bits,
			GroupSize:    groupSize,
			Mode:         mode,
		}
	}
	if useQuantized {
		slog.Warn("dequantizing expert weights: no gather kernel for format", "tensor", key, "mode", mode, "bits", bits)
	}
	return &stackedExpertWeights{
		Weight:       mlx.Dequantize(gateUp, scales, qbiases, groupSize, bits, mode, globalScale),
		Bits:         bits,
		GroupSize:    groupSize,
		Mode:         mode,
	}
}

// splitLastAxisHalves views the two halves of a's last axis.
func splitLastAxisHalves(a *mlx.Array) (lo, hi *mlx.Array) {
	dims := a.Dims()
	nd := len(dims)
	beg := make([]int32, nd)
	end := make([]int32, nd)
	for i, d := range dims {
		end[i] = int32(d)
	}
	mid := int32(dims[nd-1]) / 2
	endLo := append([]int32(nil), end...)
	endLo[nd-1] = mid
	begHi := append([]int32(nil), beg...)
	begHi[nd-1] = mid
	return mlx.SliceStartStop(a, beg, endLo), mlx.SliceStartStop(a, begHi, end)
}

// loadSwitchMLP assembles a layer's routed experts from any supported
// checkpoint layout.
func loadSwitchMLP(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, layerPrefix string) (*SwitchMLP, error) {
	// 1. Try the native fused tensor first: one tensor already covering
	//    gate+up for all experts (HF safetensors format).
	gateUpW := combinedGateUpProjection(tensors, cfg, useQuantized, layerPrefix)

	// 2. Not found -> checkpoint ships gate/up separately (mlx_lm-converted
	//    format: SwitchGLU's own gate_proj/up_proj submodule names). Load
	//    each and fuse them into one tensor at load time.
	if gateUpW == nil {
		gateW := loadStackedProjection(tensors, cfg, useQuantized,
			layerPrefix+".block_sparse_moe.switch_mlp.gate_proj")
		upW := loadStackedProjection(tensors, cfg, useQuantized,
			layerPrefix+".block_sparse_moe.switch_mlp.up_proj")
		gateUpW = fuseGateUpProjections(gateW, upW)
	}

	// Down projection: try native name first, then converted name.
	downW := loadStackedProjection(tensors, cfg, useQuantized,
		layerPrefix+".block_sparse_moe.output_linear",
		layerPrefix+".block_sparse_moe.switch_mlp.down_proj",
	)

	if gateUpW == nil || downW == nil {
		return nil, fmt.Errorf("missing switch expert weights")
	}

	// Assign to SwitchMLP fields — same quantized-vs-plain branching as old branch.
	switchMLP := &SwitchMLP{}
	if gateUpW.Scales != nil {
		switchMLP.GateUpWeightQ = gateUpW.Weight
		switchMLP.GateUpScales = gateUpW.Scales
		switchMLP.GateUpBiases = gateUpW.Biases
		switchMLP.GateUpGlobalScales = gateUpW.GlobalScales
		switchMLP.GateUpBits = gateUpW.Bits
		switchMLP.GateUpGroupSize = gateUpW.GroupSize
		switchMLP.GateUpMode = gateUpW.Mode
	} else {
		switchMLP.GateUpWeight = transposeExpertWeightForGatherMM(gateUpW.Weight)
	}
	if downW.Scales != nil {
		switchMLP.DownWeightQ = downW.Weight
		switchMLP.DownScales = downW.Scales
		switchMLP.DownBiases = downW.Biases
		switchMLP.DownGlobalScales = downW.GlobalScales
		switchMLP.DownBits = downW.Bits
		switchMLP.DownGroupSize = downW.GroupSize
		switchMLP.DownMode = downW.Mode
	} else {
		switchMLP.DownWeight = transposeExpertWeightForGatherMM(downW.Weight)
	}
	return switchMLP, nil
}

// LoadWeights receives all tensors loaded from the manifest and assigns them
// to model fields.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	m.weightPrefix = resolveWeightPrefix(tensors)
	prefix := m.weightPrefix
	cfg := m.Config
	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)

	embedTokens := model.MakeEmbeddingLayer(tensors, prefix+"model.embed_tokens", cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	if embedTokens == nil {
		return fmt.Errorf("missing embedding weight: %smodel.embed_tokens.weight", prefix)
	}
	m.EmbedTokens = embedTokens

	normWeight := tensors[prefix+"model.norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: %smodel.norm.weight", prefix)
	}
	m.Norm = nn.NewRMSNorm(normWeight, cfg.RMSNormEps)

	if m.TieWordEmbeddings {
		m.LMHead = m.EmbedTokens.AsLinear()
	} else if lmHead := linears.Make(prefix + "lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else if lmHead := linears.Make("lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else {
		m.LMHead = m.EmbedTokens.AsLinear()
	}

	for i := range m.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("%smodel.layers.%d", prefix, i)

		layer := &Layer{
			Attention: &Attention{},
		}

		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.AttentionNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.MLPNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if layer.AttentionNorm == nil {
			return fmt.Errorf("layer %d: missing input_layernorm", i)
		}
		if layer.MLPNorm == nil {
			return fmt.Errorf("layer %d: missing post_attention_layernorm", i)
		}

		layer.Attention.QProj = linears.Make(layerPrefix + ".self_attn.q_proj")
		layer.Attention.KProj = linears.Make(layerPrefix + ".self_attn.k_proj")
		layer.Attention.VProj = linears.Make(layerPrefix + ".self_attn.v_proj")
		layer.Attention.OProj = linears.Make(layerPrefix + ".self_attn.o_proj")
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			return fmt.Errorf("layer %d: missing attention projections", i)
		}

		moe := &SparseMoE{}
		moe.Router = linears.Make(layerPrefix + ".block_sparse_moe.router.layer")
		if moe.Router == nil {
			return fmt.Errorf("layer %d: missing moe router weight", i)
		}

		switchMLP, err := loadSwitchMLP(tensors, cfg, true, layerPrefix)
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
		moe.SwitchMLP = switchMLP
		layer.MoE = moe

		m.Layers[i] = layer
	}

	return nil
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))

	h := m.EmbedTokens.Forward(b.InputIDs)
	h = mlx.MulScalar(h, m.EmbeddingMultiplier)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
	}

	out := m.Norm.Forward(h, m.RMSNormEps)
	return out, out
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	logits := m.LMHead.Forward(x)
	if m.LogitsScaling != 1.0 {
		logits = mlx.DivScalar(logits, m.LogitsScaling)
	}
	return logits
}

func (m *Model) MaxContextLength() int {
	return int(m.MaxPositionEmbeddings)
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.tok
}

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i := range caches {
		caches[i] = cache.NewKVCache()
	}
	return caches
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	attnOut := l.Attention.Forward(l.AttentionNorm.Forward(x, cfg.RMSNormEps), b, c, positions, B, L, cfg)
	h := mlx.Add(x, mlx.MulScalar(attnOut, cfg.ResidualMultiplier))
	moeOut := l.MoE.Forward(l.MLPNorm.Forward(h, cfg.RMSNormEps), cfg)
	return mlx.Add(h, mlx.MulScalar(moeOut, cfg.ResidualMultiplier))
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)

	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	k = mlx.Transpose(k, 0, 2, 1, 3)

	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	// YaRN-aware RoPE: use RoPEWithFreqs when freqs are present, otherwise
	// fall back to RoPEWithBase. Apply mscale to the rotated portion.
	if cfg.RopeFreqs != nil {
		q = nn.ScaleRotaryPart(mlx.RoPEWithFreqs(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions, cfg.RopeFreqs), int(cfg.HeadDim), cfg.RopeMScale)
		k = nn.ScaleRotaryPart(mlx.RoPEWithFreqs(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions, cfg.RopeFreqs), int(cfg.HeadDim), cfg.RopeMScale)
	} else {
		q = nn.ScaleRotaryPart(mlx.RoPEWithBase(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions), int(cfg.HeadDim), cfg.RopeMScale)
		k = nn.ScaleRotaryPart(mlx.RoPEWithBase(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions), int(cfg.HeadDim), cfg.RopeMScale)
	}

	// MLX SDPA supports grouped-query attention directly (Q heads can be a
	// multiple of K/V heads), so avoid materializing repeated K/V tensors.
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

// route selects the top-k experts by raw router logits, then applies softmax
// to just the selected logits — matching GraniteMoeTopKRouter/
// GraniteMoeTopKGating. Unlike routers that softmax over all experts before
// selecting the top-k, this softmaxes only the already-truncated top-k slice,
// so no further sum-normalization is applied (or needed).
func (moe *SparseMoE) route(x *mlx.Array, cfg *Config) (inds, scores *mlx.Array) {
	logits := moe.Router.Forward(x)

	inds = mlx.Argpartition(mlx.Neg(logits), int(cfg.NumExpertsPerTok)-1, -1)
	dims := inds.Dims()
	inds = mlx.SliceStartStop(inds, []int32{0, 0, 0}, []int32{int32(dims[0]), int32(dims[1]), cfg.NumExpertsPerTok})

	selected := mlx.TakeAlongAxis(logits, inds, -1)
	scores = mlx.SoftmaxAxis(selected, -1, true)
	return inds, scores
}

func (moe *SparseMoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	inds, scores := moe.route(x, cfg)

	expertOut := moe.SwitchMLP.Forward(x, inds, cfg)
	y := mlx.Sum(mlx.Mul(expertOut, mlx.ExpandDims(scores, -1)), 2, false)

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

	var gateUp, down *mlx.Array
	if s.GateUpWeightQ != nil {
		gateUp = mlx.GatherQMM(xFlat, s.GateUpWeightQ, s.GateUpScales, s.GateUpBiases,
			nil, idxFlat, true, s.GateUpGroupSize, s.GateUpBits, s.GateUpMode,
			s.GateUpGlobalScales, doSort)
	} else {
		gateUp = mlx.GatherMM(xFlat, s.GateUpWeight, nil, idxFlat, doSort)
	}
	gate, up := splitLastAxisHalves(gateUp)
	hidden := mlx.SwiGLU(gate, up)

	if s.DownWeightQ != nil {
		down = mlx.GatherQMM(hidden, s.DownWeightQ, s.DownScales, s.DownBiases,
			nil, idxFlat, true, s.DownGroupSize, s.DownBits, s.DownMode,
			s.DownGlobalScales, doSort)
	} else {
		down = mlx.GatherMM(hidden, s.DownWeight, nil, idxFlat, doSort)
	}

	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

// Ensure Model satisfies model.Model interface.
var _ model.Model = (*Model)(nil)
