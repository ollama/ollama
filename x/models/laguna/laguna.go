// Package laguna provides the Poolside Laguna text model implementation for MLX.
package laguna

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("LagunaForCausalLM", NewModel)
}

var _ base.Model = (*Model)(nil)

type gatingMode string

type ropeConfig struct {
	flat    *nn.RopeParameters
	full    *nn.RopeParameters
	sliding *nn.RopeParameters
	nested  bool
}

type Config struct {
	ModelType                   string             `json:"model_type"`
	HiddenSize                  int32              `json:"hidden_size"`
	IntermediateSize            int32              `json:"intermediate_size"`
	MoeIntermediateSize         int32              `json:"moe_intermediate_size"`
	SharedExpertIntermediate    int32              `json:"shared_expert_intermediate_size"`
	NumHiddenLayers             int32              `json:"num_hidden_layers"`
	NumAttentionHeads           int32              `json:"num_attention_heads"`
	NumAttentionHeadsPerLayer   []int32            `json:"num_attention_heads_per_layer"`
	NumKeyValueHeads            int32              `json:"num_key_value_heads"`
	HeadDim                     int32              `json:"head_dim"`
	RMSNormEps                  float32            `json:"rms_norm_eps"`
	VocabSize                   int32              `json:"vocab_size"`
	MaxPositionEmbeddings       int32              `json:"max_position_embeddings"`
	LayerTypes                  []string           `json:"layer_types"`
	SlidingWindow               int32              `json:"sliding_window"`
	MLPOnlyLayers               []int32            `json:"mlp_only_layers"`
	DecoderSparseStep           int32              `json:"decoder_sparse_step"`
	NumExperts                  int32              `json:"num_experts"`
	NumExpertsPerTok            int32              `json:"num_experts_per_tok"`
	NormTopKProb                bool               `json:"norm_topk_prob"`
	MoeRoutedScalingFactor      float32            `json:"moe_routed_scaling_factor"`
	MoeApplyRouterWeightOnInput bool               `json:"moe_apply_router_weight_on_input"`
	Gating                      string             `json:"gating"`
	TieWordEmbeddings           bool               `json:"tie_word_embeddings"`
	RopeTheta                   float32            `json:"rope_theta"`
	PartialRotaryFactor         float32            `json:"partial_rotary_factor"`
	RopeParameters              *nn.RopeParameters `json:"rope_parameters"`
	RopeScaling                 *nn.RopeParameters `json:"rope_scaling"`
	SWARopeParameters           *nn.RopeParameters `json:"swa_rope_parameters"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	Scale            float32    `json:"-"`
	FullRopeDim      int        `json:"-"`
	FullRopeBase     float32    `json:"-"`
	FullRopeScale    float32    `json:"-"`
	FullRopeFreqs    *mlx.Array `json:"-"`
	SlidingRopeDim   int        `json:"-"`
	SlidingRopeBase  float32    `json:"-"`
	SlidingRopeScale float32    `json:"-"`
}

type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config
}

type Layer struct {
	InputNorm         *nn.RMSNorm
	PostAttentionNorm *nn.RMSNorm
	Attention         *Attention
	MLP               MLPBlock

	LayerIdx  int32
	IsSliding bool
}

type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
	GProj nn.LinearLayer

	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm

	NumHeads int32
}

type MLPBlock interface {
	Forward(x *mlx.Array, cfg *Config) *mlx.Array
}

type MLPBlockAdd interface {
	ForwardAdd(x, residual *mlx.Array, cfg *Config) *mlx.Array
}

type DenseMLP struct {
	GateProj        nn.LinearLayer
	UpProj          nn.LinearLayer
	DownProj        nn.LinearLayer
	GateUpProj      nn.LinearLayer
	GateUpGateScale *mlx.Array
	GateUpUpScale   *mlx.Array
}

type lagunaQuantizedLinear struct {
	*nn.QuantizedLinear
}

type SparseMoE struct {
	Gate                 nn.LinearLayer
	SwitchMLP            *SwitchMLP
	SharedExpert         *DenseMLP
	EScoreCorrectionBias *mlx.Array
}

type SwitchMLP struct {
	GateUpWeight *mlx.Array
	GateWeight   *mlx.Array
	UpWeight     *mlx.Array
	DownWeight   *mlx.Array
	// DenseWeightsSourceLayout means dense expert weights are stored as
	// [experts, out, in], matching the published tensors. GatherMM fallback
	// transposes lazily so load does not materialize huge BF16 expert tensors.
	DenseWeightsSourceLayout bool

	GateUpWeightQ, GateUpScales, GateUpBiases *mlx.Array
	GateWeightQ, GateScales, GateBiases       *mlx.Array
	UpWeightQ, UpScales, UpBiases             *mlx.Array
	DownWeightQ, DownScales, DownBiases       *mlx.Array
	GateGlobalScale, UpGlobalScale            *mlx.Array
	DownGlobalScale                           *mlx.Array

	GateUpBits, GateBits, UpBits, DownBits                     int
	GateUpGroupSize, GateGroupSize, UpGroupSize, DownGroupSize int
	GateUpMode, GateMode, UpMode, DownMode                     string
	UseQuantized, UseFusedGateUp                               bool
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

func parseConfig(configData []byte) (Config, error) {
	type rawConfig struct {
		ModelType                   string             `json:"model_type"`
		HiddenSize                  int32              `json:"hidden_size"`
		IntermediateSize            int32              `json:"intermediate_size"`
		MoeIntermediateSize         int32              `json:"moe_intermediate_size"`
		SharedExpertIntermediate    int32              `json:"shared_expert_intermediate_size"`
		NumHiddenLayers             int32              `json:"num_hidden_layers"`
		NumAttentionHeads           int32              `json:"num_attention_heads"`
		NumAttentionHeadsPerLayer   []int32            `json:"num_attention_heads_per_layer"`
		NumKeyValueHeads            int32              `json:"num_key_value_heads"`
		HeadDim                     int32              `json:"head_dim"`
		RMSNormEps                  float32            `json:"rms_norm_eps"`
		VocabSize                   int32              `json:"vocab_size"`
		MaxPositionEmbeddings       int32              `json:"max_position_embeddings"`
		LayerTypes                  []string           `json:"layer_types"`
		SlidingWindow               int32              `json:"sliding_window"`
		MLPOnlyLayers               []int32            `json:"mlp_only_layers"`
		MLPLayerTypes               []string           `json:"mlp_layer_types"`
		DecoderSparseStep           int32              `json:"decoder_sparse_step"`
		NumExperts                  int32              `json:"num_experts"`
		NumExpertsPerTok            int32              `json:"num_experts_per_tok"`
		NormTopKProb                *bool              `json:"norm_topk_prob"`
		MoeRoutedScalingFactor      float32            `json:"moe_routed_scaling_factor"`
		MoeApplyRouterWeightOnInput bool               `json:"moe_apply_router_weight_on_input"`
		Gating                      gatingMode         `json:"gating"`
		TieWordEmbeddings           bool               `json:"tie_word_embeddings"`
		RopeTheta                   float32            `json:"rope_theta"`
		PartialRotaryFactor         float32            `json:"partial_rotary_factor"`
		RopeParameters              ropeConfig         `json:"rope_parameters"`
		RopeScaling                 *nn.RopeParameters `json:"rope_scaling"`
		SWARopeParameters           *nn.RopeParameters `json:"swa_rope_parameters"`
	}

	var raw rawConfig
	if err := json.Unmarshal(configData, &raw); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}

	mlpOnlyLayers, err := denseLayers(raw.MLPOnlyLayers, raw.MLPLayerTypes)
	if err != nil {
		return Config{}, err
	}

	fullRope := raw.RopeParameters.fullParams()
	if fullRope == nil {
		fullRope = raw.RopeScaling
	}
	swaRope := raw.SWARopeParameters
	if nestedSwa := raw.RopeParameters.slidingParams(); nestedSwa != nil {
		swaRope = nestedSwa
	}

	cfg := Config{
		ModelType:                   raw.ModelType,
		HiddenSize:                  raw.HiddenSize,
		IntermediateSize:            raw.IntermediateSize,
		MoeIntermediateSize:         raw.MoeIntermediateSize,
		SharedExpertIntermediate:    raw.SharedExpertIntermediate,
		NumHiddenLayers:             raw.NumHiddenLayers,
		NumAttentionHeads:           raw.NumAttentionHeads,
		NumAttentionHeadsPerLayer:   raw.NumAttentionHeadsPerLayer,
		NumKeyValueHeads:            raw.NumKeyValueHeads,
		HeadDim:                     raw.HeadDim,
		RMSNormEps:                  raw.RMSNormEps,
		VocabSize:                   raw.VocabSize,
		MaxPositionEmbeddings:       raw.MaxPositionEmbeddings,
		LayerTypes:                  raw.LayerTypes,
		SlidingWindow:               raw.SlidingWindow,
		MLPOnlyLayers:               mlpOnlyLayers,
		DecoderSparseStep:           raw.DecoderSparseStep,
		NumExperts:                  raw.NumExperts,
		NumExpertsPerTok:            raw.NumExpertsPerTok,
		NormTopKProb:                defaultBool(raw.NormTopKProb, true),
		MoeRoutedScalingFactor:      raw.MoeRoutedScalingFactor,
		MoeApplyRouterWeightOnInput: raw.MoeApplyRouterWeightOnInput,
		Gating:                      raw.Gating.normalized(),
		TieWordEmbeddings:           raw.TieWordEmbeddings,
		RopeTheta:                   raw.RopeTheta,
		PartialRotaryFactor:         raw.PartialRotaryFactor,
		RopeParameters:              fullRope,
		RopeScaling:                 raw.RopeScaling,
		SWARopeParameters:           swaRope,
	}

	if cfg.HiddenSize <= 0 {
		return Config{}, fmt.Errorf("invalid hidden_size: %d", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers <= 0 {
		return Config{}, fmt.Errorf("invalid num_hidden_layers: %d", cfg.NumHiddenLayers)
	}
	if cfg.NumAttentionHeads <= 0 && len(cfg.NumAttentionHeadsPerLayer) == 0 {
		return Config{}, fmt.Errorf("missing num_attention_heads")
	}
	if cfg.NumKeyValueHeads <= 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HeadDim <= 0 {
		if cfg.NumAttentionHeads <= 0 || cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
			return Config{}, fmt.Errorf("cannot infer head_dim")
		}
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.IntermediateSize <= 0 {
		return Config{}, fmt.Errorf("invalid intermediate_size: %d", cfg.IntermediateSize)
	}
	if cfg.MoeIntermediateSize <= 0 {
		cfg.MoeIntermediateSize = cfg.IntermediateSize
	}
	if cfg.SharedExpertIntermediate <= 0 {
		cfg.SharedExpertIntermediate = cfg.MoeIntermediateSize
	}
	if cfg.DecoderSparseStep <= 0 {
		cfg.DecoderSparseStep = 1
	}
	if cfg.NumExpertsPerTok <= 0 && cfg.NumExperts > 0 {
		cfg.NumExpertsPerTok = 1
	}
	if cfg.MoeRoutedScalingFactor == 0 {
		cfg.MoeRoutedScalingFactor = 1
	}

	ropeParams := cfg.RopeParameters
	if ropeParams == nil {
		ropeParams = cfg.RopeScaling
	}
	cfg.FullRopeBase = cfg.RopeTheta
	if cfg.FullRopeBase == 0 && ropeParams != nil && ropeParams.RopeTheta > 0 {
		cfg.FullRopeBase = ropeParams.RopeTheta
	}
	if cfg.FullRopeBase == 0 {
		cfg.FullRopeBase = 10000
	}
	fullPartial := cfg.PartialRotaryFactor
	if fullPartial == 0 && ropeParams != nil && ropeParams.PartialRotaryFactor > 0 {
		fullPartial = ropeParams.PartialRotaryFactor
	}
	if fullPartial == 0 {
		fullPartial = 1
	}
	cfg.FullRopeDim = clampRopeDim(int(float32(cfg.HeadDim)*fullPartial), int(cfg.HeadDim))
	cfg.FullRopeScale = 1
	if ropeParams != nil && strings.EqualFold(ropeParams.TypeName(), "yarn") {
		cfg.FullRopeFreqs, cfg.FullRopeScale = nn.BuildYarnRopeFreqs(cfg.FullRopeDim, cfg.FullRopeBase, ropeParams)
	}

	cfg.SlidingRopeBase = cfg.FullRopeBase
	slidingPartial := fullPartial
	if cfg.SWARopeParameters != nil {
		if cfg.SWARopeParameters.RopeTheta > 0 {
			cfg.SlidingRopeBase = cfg.SWARopeParameters.RopeTheta
		}
		if cfg.SWARopeParameters.PartialRotaryFactor > 0 {
			slidingPartial = cfg.SWARopeParameters.PartialRotaryFactor
		}
	}
	cfg.SlidingRopeDim = clampRopeDim(int(float32(cfg.HeadDim)*slidingPartial), int(cfg.HeadDim))
	cfg.SlidingRopeScale = 1
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	return cfg, nil
}

func (g *gatingMode) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		*g = gatingMode(s)
		return nil
	}

	var enabled bool
	if err := json.Unmarshal(b, &enabled); err == nil {
		if enabled {
			*g = "per-head"
		} else {
			*g = "false"
		}
		return nil
	}

	if string(b) == "null" {
		return nil
	}
	return fmt.Errorf("unsupported Laguna gating JSON value %s", string(b))
}

func (g gatingMode) normalized() string {
	if strings.EqualFold(string(g), "true") {
		return "per-head"
	}
	return string(g)
}

func (r *ropeConfig) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		return nil
	}

	var probe map[string]json.RawMessage
	if err := json.Unmarshal(b, &probe); err != nil {
		return err
	}
	if len(probe) == 0 {
		return nil
	}

	if raw, ok := probe["full_attention"]; ok {
		r.nested = true
		r.full = &nn.RopeParameters{}
		if err := json.Unmarshal(raw, r.full); err != nil {
			return err
		}
		if raw = probe["sliding_attention"]; raw != nil {
			r.sliding = &nn.RopeParameters{}
			if err := json.Unmarshal(raw, r.sliding); err != nil {
				return err
			}
		}
		return nil
	}

	if raw, ok := probe["global_attention"]; ok {
		r.nested = true
		r.full = &nn.RopeParameters{}
		if err := json.Unmarshal(raw, r.full); err != nil {
			return err
		}
		if raw = probe["sliding_attention"]; raw != nil {
			r.sliding = &nn.RopeParameters{}
			if err := json.Unmarshal(raw, r.sliding); err != nil {
				return err
			}
		}
		return nil
	}

	r.flat = &nn.RopeParameters{}
	return json.Unmarshal(b, r.flat)
}

func (r ropeConfig) fullParams() *nn.RopeParameters {
	if r.nested {
		return r.full
	}
	return r.flat
}

func (r ropeConfig) slidingParams() *nn.RopeParameters {
	if !r.nested {
		return nil
	}
	return r.sliding
}

func defaultBool(v *bool, fallback bool) bool {
	if v == nil {
		return fallback
	}
	return *v
}

func denseLayers(mlpOnlyLayers []int32, mlpLayerTypes []string) ([]int32, error) {
	if len(mlpOnlyLayers) > 0 {
		return mlpOnlyLayers, nil
	}
	if len(mlpLayerTypes) == 0 {
		return nil, nil
	}

	dense := make([]int32, 0, len(mlpLayerTypes))
	for i, layerType := range mlpLayerTypes {
		switch {
		case strings.EqualFold(layerType, "dense"):
			dense = append(dense, int32(i))
		case strings.EqualFold(layerType, "sparse"):
		default:
			return nil, fmt.Errorf("unsupported mlp_layer_types[%d]=%q", i, layerType)
		}
	}
	return dense, nil
}

func clampRopeDim(v, maxDim int) int {
	if v <= 0 {
		return maxDim
	}
	if v > maxDim {
		return maxDim
	}
	if v%2 != 0 {
		v--
	}
	if v <= 0 {
		return maxDim
	}
	return v
}

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
		m.Layers[i] = &Layer{LayerIdx: i, IsSliding: layerIsSliding(&cfg, i)}
	}
	return m, nil
}

func layerIsSliding(cfg *Config, layer int32) bool {
	if len(cfg.LayerTypes) == int(cfg.NumHiddenLayers) {
		return strings.EqualFold(cfg.LayerTypes[layer], "sliding_attention")
	}
	return false
}

func layerUsesMoE(cfg *Config, layer int32) bool {
	if cfg.NumExperts <= 0 {
		return false
	}
	for _, l := range cfg.MLPOnlyLayers {
		if l == layer {
			return false
		}
	}
	return (layer+1)%cfg.DecoderSparseStep == 0
}

func numHeadsForLayer(cfg *Config, layer int32) int32 {
	if int(layer) < len(cfg.NumAttentionHeadsPerLayer) && cfg.NumAttentionHeadsPerLayer[layer] > 0 {
		return cfg.NumAttentionHeadsPerLayer[layer]
	}
	return cfg.NumAttentionHeads
}

func resolveWeightPrefix(tensors map[string]*mlx.Array) string {
	for _, prefix := range []string{"model.", "", "language_model.model.", "language_model.", "model.language_model.model.", "model.language_model."} {
		if tensors[prefix+"embed_tokens.weight"] != nil {
			return prefix
		}
	}
	return "model."
}

func tensorAny(tensors map[string]*mlx.Array, keys ...string) (*mlx.Array, string) {
	for _, k := range keys {
		if v := tensors[k]; v != nil {
			return v, k
		}
	}
	return nil, ""
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

func freeTensorKeys(tensors map[string]*mlx.Array, keys ...string) {
	for _, k := range keys {
		if k == "" {
			continue
		}
		if t := tensors[k]; t != nil {
			delete(tensors, k)
		}
	}
}

func stackAndClone(parts []*mlx.Array) *mlx.Array {
	if len(parts) == 0 {
		return nil
	}
	stacked := mlx.Stack(parts, 0).Clone()
	mlx.Eval(stacked)
	return stacked
}

func transposeExpertWeightForGatherMM(w *mlx.Array) *mlx.Array {
	if w == nil || !w.Valid() || w.NumDims() != 3 {
		return w
	}
	t := mlx.Transpose(w, 0, 2, 1).Clone()
	mlx.Eval(t)
	return t
}

func transposeExpertWeightViewForGatherMM(w *mlx.Array) *mlx.Array {
	if w == nil || !w.Valid() || w.NumDims() != 3 {
		return w
	}
	return mlx.Transpose(w, 0, 2, 1)
}

func denseExpertWeight(w *stackedExpertWeights) *mlx.Array {
	if w == nil {
		return nil
	}
	weight := w.Weight
	if w.Scales != nil {
		weight = mlx.Dequantize(w.Weight, w.Scales, w.Biases, w.GroupSize, w.Bits, w.Mode)
		if w.GlobalScales != nil {
			scale := w.GlobalScales
			if scale.DType() != weight.DType() {
				scale = scale.AsType(weight.DType())
			}
			if !(scale.NumDims() == 0 || (scale.NumDims() == 1 && scale.Dim(0) == 1)) {
				scale = mlx.ExpandDims(mlx.ExpandDims(scale, -1), -1)
			}
			weight = mlx.Mul(weight, scale)
		}
	}
	return weight
}

func denseExpertWeightForGatherMM(w *stackedExpertWeights) *mlx.Array {
	weight := denseExpertWeight(w)
	if weight == nil {
		return nil
	}
	return transposeExpertWeightForGatherMM(weight)
}

func denseExpertWeightSupportsSourceLayout(w *stackedExpertWeights) bool {
	return w != nil && w.Weight != nil && w.Weight.Valid() && w.Scales == nil && w.Weight.DType() == mlx.DTypeBFloat16
}

func denseExpertWeightsSupportSourceLayout(weights ...*stackedExpertWeights) bool {
	for _, w := range weights {
		if !denseExpertWeightSupportsSourceLayout(w) {
			return false
		}
	}
	return true
}

func canFuseQuantizedGateUp(gateW, upW *stackedExpertWeights) bool {
	if gateW == nil || upW == nil || gateW.Scales == nil || upW.Scales == nil {
		return false
	}
	if gateW.GlobalScales != nil || upW.GlobalScales != nil {
		return false
	}
	if gateW.Bits != upW.Bits || gateW.GroupSize != upW.GroupSize || gateW.Mode != upW.Mode {
		return false
	}
	if (gateW.Biases == nil) != (upW.Biases == nil) {
		return false
	}
	return gateW.Weight.NumDims() == 3 && upW.Weight.NumDims() == 3
}

func canFuseDenseQuantizedLinears(a, b *nn.QuantizedLinear) bool {
	if a == nil || b == nil || a.Scales == nil || b.Scales == nil {
		return false
	}
	if a.QBiases != nil || b.QBiases != nil ||
		a.Bias != nil || b.Bias != nil {
		return false
	}
	if !isScalarGlobalScale(a.GlobalScale) || !isScalarGlobalScale(b.GlobalScale) {
		return false
	}
	if a.Bits != b.Bits || a.GroupSize != b.GroupSize || a.Mode != b.Mode {
		return false
	}
	if a.Weight.NumDims() != 2 || b.Weight.NumDims() != 2 ||
		a.Scales.NumDims() != 2 || b.Scales.NumDims() != 2 {
		return false
	}
	if a.Weight.Dim(1) != b.Weight.Dim(1) || a.Scales.Dim(1) != b.Scales.Dim(1) {
		return false
	}
	if a.Mode == "nvfp4" && a.Bits == 4 && a.GroupSize == 16 {
		return a.Weight.Dim(0)+b.Weight.Dim(0) >= 512 && a.Weight.Dim(1)*8 >= 512
	}
	if a.Mode == "mxfp8" && a.Bits == 8 && a.GroupSize == 32 {
		return a.Weight.Dim(0)+b.Weight.Dim(0) >= 512 && a.Weight.Dim(1)*4 >= 512
	}
	return false
}

func isScalarGlobalScale(scale *mlx.Array) bool {
	if scale == nil {
		return true
	}
	return scale.NumDims() == 0 || (scale.NumDims() == 1 && scale.Dim(0) == 1)
}

func wrapLagunaLinear(l nn.LinearLayer) nn.LinearLayer {
	q, ok := l.(*nn.QuantizedLinear)
	if !ok {
		return l
	}
	if q.Scales == nil || q.QBiases != nil || q.Bias != nil {
		return l
	}
	if q.Mode == "nvfp4" && q.Bits == 4 && q.GroupSize == 16 && isScalarGlobalScale(q.GlobalScale) {
		return &lagunaQuantizedLinear{QuantizedLinear: q}
	}
	if q.Mode == "mxfp8" && q.Bits == 8 && q.GroupSize == 32 && q.GlobalScale == nil {
		return &lagunaQuantizedLinear{QuantizedLinear: q}
	}
	return l
}

func (l *lagunaQuantizedLinear) Forward(x *mlx.Array) *mlx.Array {
	if x.NumDims() == 3 && x.Dim(0) == 1 && x.Dim(1) >= 64 {
		if y, ok := mlx.FastQuantizedLinear(x, l.Weight, l.Scales, l.GlobalScale, l.GroupSize, l.Bits, l.Mode); ok {
			return y
		}
	}
	return l.QuantizedLinear.Forward(x)
}

func fuseDenseQuantizedLinears(a, b nn.LinearLayer) nn.LinearLayer {
	aq, ok := a.(*lagunaQuantizedLinear)
	if !ok {
		if q, ok := a.(*nn.QuantizedLinear); ok {
			aq = &lagunaQuantizedLinear{QuantizedLinear: q}
		} else {
			return nil
		}
	}
	bq, ok := b.(*lagunaQuantizedLinear)
	if !ok {
		if q, ok := b.(*nn.QuantizedLinear); ok {
			bq = &lagunaQuantizedLinear{QuantizedLinear: q}
		} else {
			return nil
		}
	}
	if !canFuseDenseQuantizedLinears(aq.QuantizedLinear, bq.QuantizedLinear) {
		return nil
	}
	return &lagunaQuantizedLinear{QuantizedLinear: &nn.QuantizedLinear{
		Weight:    fuseExpertStacks(aq.Weight, bq.Weight, 0),
		Scales:    fuseExpertStacks(aq.Scales, bq.Scales, 0),
		GroupSize: aq.GroupSize,
		Bits:      aq.Bits,
		Mode:      aq.Mode,
	}}
}

// fuseDenseGateUp fuses only the quantized weights and scales. Scalar global
// scales from the original projections stay on DenseMLP and are applied after
// the fused output is split back into gate/up halves.
func fuseDenseGateUp(gate, up nn.LinearLayer) nn.LinearLayer {
	return fuseDenseQuantizedLinears(gate, up)
}

func linearGlobalScale(l nn.LinearLayer) *mlx.Array {
	switch q := l.(type) {
	case *lagunaQuantizedLinear:
		return q.GlobalScale
	case *nn.QuantizedLinear:
		return q.GlobalScale
	default:
		return nil
	}
}

func applyDenseGlobalScale(x, globalScale *mlx.Array) *mlx.Array {
	if globalScale == nil {
		return x
	}
	return mlx.Mul(x, globalScale).AsType(x.DType())
}

func splitLastDim(x *mlx.Array, first int32) (*mlx.Array, *mlx.Array) {
	dims := x.Dims()
	starts := make([]int32, len(dims))
	leftStops := make([]int32, len(dims))
	rightStarts := make([]int32, len(dims))
	rightStops := make([]int32, len(dims))
	for i, dim := range dims {
		leftStops[i] = int32(dim)
		rightStops[i] = int32(dim)
	}
	leftStops[len(leftStops)-1] = first
	rightStarts[len(rightStarts)-1] = first
	return mlx.SliceStartStop(x, starts, leftStops), mlx.SliceStartStop(x, rightStarts, rightStops)
}

func shouldMapMoEExperts(tokens int32) bool {
	return tokens >= 64
}

func shouldMapDenseMoEExperts(tokens int32) bool {
	// Dense mapped MM is gated separately by dtype/backend support. When it is
	// available, use it for both prompt and decode because Laguna BF16 otherwise
	// spends heavily in generic GatherMM at all batch sizes.
	return tokens > 0
}

func fuseExpertStacks(a, b *mlx.Array, axis int) *mlx.Array {
	if a == nil || !a.Valid() || b == nil || !b.Valid() {
		return nil
	}
	out := mlx.Concatenate([]*mlx.Array{a, b}, axis).Clone()
	mlx.Eval(out)
	return out
}

func applyExpertGlobalScale(x, globalScale, idx *mlx.Array) *mlx.Array {
	if globalScale == nil {
		return x
	}
	if globalScale.NumDims() == 0 || (globalScale.NumDims() == 1 && globalScale.Dim(0) == 1) {
		return mlx.Mul(x, globalScale).AsType(x.DType())
	}
	scale := mlx.ExpandDims(mlx.ExpandDims(mlx.Take(globalScale, idx, 0), -1), -1)
	return mlx.Mul(x, scale).AsType(x.DType())
}

func combinedTensorGlobalScale(tensors map[string]*mlx.Array, key string) (*mlx.Array, []string) {
	var names []string
	weightGlobal := tensors[key+".global_scale"]
	if weightGlobal == nil {
		weightGlobal = tensors[key+".weight.global_scale"]
	}
	if weightGlobal != nil {
		names = append(names, key+".global_scale", key+".weight.global_scale")
	}
	if tensors[key+".input_global_scale"] != nil || tensors[key+".weight.input_global_scale"] != nil {
		names = append(names, key+".input_global_scale", key+".weight.input_global_scale")
	}
	switch {
	case weightGlobal != nil:
		return weightGlobal, names
	default:
		return nil, nil
	}
}

func collectPerExpertProjection(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, layerPrefix, proj string, numExperts int32) *stackedExpertWeights {
	weights := make([]*mlx.Array, 0, numExperts)
	scales := make([]*mlx.Array, 0, numExperts)
	biases := make([]*mlx.Array, 0, numExperts)
	globalScales := make([]*mlx.Array, 0, numExperts)
	consumedKeys := make([]string, 0, numExperts*5)
	bits := 0
	groupSize := 0
	mode := cfg.QuantMode

	for e := range numExperts {
		base := fmt.Sprintf("%s.mlp.experts.%d.%s", layerPrefix, e, proj)
		w, key := tensorAny(tensors, base+".weight", base)
		if w == nil {
			return nil
		}
		consumedKeys = append(consumedKeys, key)
		s := tensors[key+"_scale"]
		if s == nil {
			s = tensors[key+".scale"]
		}
		if s == nil {
			weights = append(weights, w)
			continue
		}
		consumedKeys = append(consumedKeys, key+"_scale", key+".scale")
		qb := tensors[key+"_qbias"]
		if qb == nil {
			qb = tensors[key+".bias"]
		}
		if qb != nil {
			consumedKeys = append(consumedKeys, key+"_qbias", key+".bias")
		}
		globalScale, globalScaleKeys := combinedTensorGlobalScale(tensors, key)
		if globalScale != nil {
			consumedKeys = append(consumedKeys, globalScaleKeys...)
		}
		gs, b, m := model.ResolveLinearQuantParams(cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant, key, w, s)
		if bits == 0 {
			bits = b
			groupSize = gs
			mode = m
		}
		if useQuantized && supportsGatherQMM(m, b) {
			weights = append(weights, w)
			scales = append(scales, s)
			if globalScale != nil {
				globalScales = append(globalScales, globalScale)
			}
			if qb != nil {
				biases = append(biases, qb)
			}
		} else {
			deq := mlx.Dequantize(w, s, qb, gs, b, m)
			if globalScale != nil {
				deq = mlx.Mul(deq, globalScale)
				globalScales = append(globalScales, globalScale)
			}
			weights = append(weights, deq)
		}
	}

	out := &stackedExpertWeights{Weight: stackAndClone(weights), Bits: bits, GroupSize: groupSize, Mode: mode}
	if len(scales) == len(weights) {
		out.Scales = stackAndClone(scales)
	}
	if len(biases) == len(weights) {
		out.Biases = stackAndClone(biases)
	}
	if len(globalScales) == len(weights) {
		out.GlobalScales = stackAndClone(globalScales)
	}
	freeTensorKeys(tensors, consumedKeys...)
	return out
}

func loadStackedProjection(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, bases ...string) *stackedExpertWeights {
	for _, base := range bases {
		w, key := tensorAny(tensors, base+".weight", base)
		if w == nil {
			continue
		}
		consumedKeys := []string{key}
		s := tensors[key+"_scale"]
		if s != nil {
			consumedKeys = append(consumedKeys, key+"_scale")
		}
		if s == nil {
			s = tensors[key+".scale"]
			if s != nil {
				consumedKeys = append(consumedKeys, key+".scale")
			}
		}
		if s == nil {
			freeTensorKeys(tensors, consumedKeys...)
			return &stackedExpertWeights{Weight: w}
		}
		qb := tensors[key+"_qbias"]
		if qb != nil {
			consumedKeys = append(consumedKeys, key+"_qbias")
		}
		if qb == nil {
			qb = tensors[key+".bias"]
			if qb != nil {
				consumedKeys = append(consumedKeys, key+".bias")
			}
		}
		globalScale, globalScaleKeys := combinedTensorGlobalScale(tensors, key)
		consumedKeys = append(consumedKeys, globalScaleKeys...)
		gs, b, m := model.ResolveLinearQuantParams(cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant, key, w, s)
		if useQuantized && supportsGatherQMM(m, b) {
			freeTensorKeys(tensors, consumedKeys...)
			return &stackedExpertWeights{Weight: w, Scales: s, Biases: qb, GlobalScales: globalScale, Bits: b, GroupSize: gs, Mode: m}
		}
		deq := mlx.Dequantize(w, s, qb, gs, b, m)
		if globalScale != nil {
			deq = mlx.Mul(deq, globalScale)
		}
		freeTensorKeys(tensors, consumedKeys...)
		return &stackedExpertWeights{Weight: deq, GlobalScales: globalScale, Bits: b, GroupSize: gs, Mode: m}
	}
	return nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	prefix := resolveWeightPrefix(tensors)
	cfg := m.Config
	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	makeLinear := func(name string) nn.LinearLayer {
		return wrapLagunaLinear(linears.Make(name))
	}

	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, prefix+"embed_tokens", cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing embedding weight: %sembed_tokens.weight", prefix)
	}
	if w := tensors[prefix+"norm.weight"]; w != nil {
		m.Norm = nn.NewRMSNorm(w, cfg.RMSNormEps)
	} else {
		return fmt.Errorf("missing final norm weight: %snorm.weight", prefix)
	}
	if cfg.TieWordEmbeddings {
		m.LMHead = wrapLagunaLinear(m.EmbedTokens.AsLinear())
	} else if lmHead := makeLinear("lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else if lmHead := makeLinear(prefix + "lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else {
		return fmt.Errorf("missing lm_head.weight")
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
		layerPrefix := fmt.Sprintf("%slayers.%d", prefix, i)
		layer := &Layer{
			LayerIdx:  i,
			IsSliding: layerIsSliding(cfg, i),
			Attention: &Attention{NumHeads: numHeadsForLayer(cfg, i)},
		}
		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.InputNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.PostAttentionNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if layer.InputNorm == nil || layer.PostAttentionNorm == nil {
			return fmt.Errorf("layer %d: missing layer norms", i)
		}

		layer.Attention.QProj = makeLinear(layerPrefix + ".self_attn.q_proj")
		layer.Attention.KProj = makeLinear(layerPrefix + ".self_attn.k_proj")
		layer.Attention.VProj = makeLinear(layerPrefix + ".self_attn.v_proj")
		layer.Attention.OProj = makeLinear(layerPrefix + ".self_attn.o_proj")
		layer.Attention.GProj = makeLinear(layerPrefix + ".self_attn.g_proj")
		if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
			layer.Attention.QNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.k_norm.weight"]; w != nil {
			layer.Attention.KNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil || layer.Attention.GProj == nil {
			return fmt.Errorf("layer %d: missing attention projections", i)
		}
		if layer.Attention.QNorm == nil || layer.Attention.KNorm == nil {
			return fmt.Errorf("layer %d: missing attention q/k norms", i)
		}

		if layerUsesMoE(cfg, i) {
			moe := &SparseMoE{Gate: makeLinear(layerPrefix + ".mlp.gate")}
			if moe.Gate == nil {
				return fmt.Errorf("layer %d: missing moe gate", i)
			}
			moe.EScoreCorrectionBias, _ = tensorAny(tensors,
				layerPrefix+".mlp.experts.e_score_correction_bias",
				layerPrefix+".mlp.switch_mlp.e_score_correction_bias",
			)
			if moe.EScoreCorrectionBias != nil && moe.EScoreCorrectionBias.DType() != mlx.DTypeFloat32 {
				bias := moe.EScoreCorrectionBias.AsType(mlx.DTypeFloat32).Clone()
				mlx.Eval(bias)
				moe.EScoreCorrectionBias = bias
			}

			gateW := loadStackedProjection(tensors, cfg, useQuantizedExperts,
				layerPrefix+".mlp.switch_mlp.gate_proj",
				layerPrefix+".mlp.experts.gate_proj",
			)
			upW := loadStackedProjection(tensors, cfg, useQuantizedExperts,
				layerPrefix+".mlp.switch_mlp.up_proj",
				layerPrefix+".mlp.experts.up_proj",
			)
			downW := loadStackedProjection(tensors, cfg, useQuantizedExperts,
				layerPrefix+".mlp.switch_mlp.down_proj",
				layerPrefix+".mlp.experts.down_proj",
			)
			if gateW == nil || upW == nil || downW == nil {
				gateW = collectPerExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "gate_proj", cfg.NumExperts)
				upW = collectPerExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "up_proj", cfg.NumExperts)
				downW = collectPerExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "down_proj", cfg.NumExperts)
			}
			if gateW == nil || upW == nil || downW == nil {
				return fmt.Errorf("layer %d: missing moe expert weights", i)
			}
			sw := &SwitchMLP{}
			if gateW.Scales != nil && upW.Scales != nil && downW.Scales != nil {
				sw.UseQuantized = true
				sw.DownWeightQ, sw.DownScales, sw.DownBiases = downW.Weight, downW.Scales, downW.Biases
				sw.DownGlobalScale = downW.GlobalScales
				sw.DownBits, sw.DownGroupSize, sw.DownMode = downW.Bits, downW.GroupSize, downW.Mode
				if canFuseQuantizedGateUp(gateW, upW) {
					sw.UseFusedGateUp = true
					sw.GateUpWeightQ = fuseExpertStacks(gateW.Weight, upW.Weight, 1)
					sw.GateUpScales = fuseExpertStacks(gateW.Scales, upW.Scales, 1)
					sw.GateUpBiases = fuseExpertStacks(gateW.Biases, upW.Biases, 1)
					sw.GateUpBits, sw.GateUpGroupSize, sw.GateUpMode = gateW.Bits, gateW.GroupSize, gateW.Mode
				} else {
					sw.GateWeightQ, sw.GateScales, sw.GateBiases = gateW.Weight, gateW.Scales, gateW.Biases
					sw.UpWeightQ, sw.UpScales, sw.UpBiases = upW.Weight, upW.Scales, upW.Biases
					sw.GateGlobalScale = gateW.GlobalScales
					sw.UpGlobalScale = upW.GlobalScales
					sw.GateBits, sw.GateGroupSize, sw.GateMode = gateW.Bits, gateW.GroupSize, gateW.Mode
					sw.UpBits, sw.UpGroupSize, sw.UpMode = upW.Bits, upW.GroupSize, upW.Mode
				}
			} else {
				sw.DenseWeightsSourceLayout = denseExpertWeightsSupportSourceLayout(gateW, upW, downW)
				if sw.DenseWeightsSourceLayout {
					sw.GateWeight = denseExpertWeight(gateW)
					sw.UpWeight = denseExpertWeight(upW)
					sw.DownWeight = denseExpertWeight(downW)
					// Avoid pre-fusing source-layout BF16 gate/up weights: the
					// full-size concatenate can time out during model load.
				} else {
					sw.GateWeight = denseExpertWeightForGatherMM(gateW)
					sw.UpWeight = denseExpertWeightForGatherMM(upW)
					sw.DownWeight = denseExpertWeightForGatherMM(downW)
					sw.GateUpWeight = fuseExpertStacks(sw.GateWeight, sw.UpWeight, 2)
				}
				sw.UseFusedGateUp = sw.GateUpWeight != nil
			}
			moe.SwitchMLP = sw
			sharedGate := makeLinear(layerPrefix + ".mlp.shared_expert.gate_proj")
			sharedUp := makeLinear(layerPrefix + ".mlp.shared_expert.up_proj")
			moe.SharedExpert = &DenseMLP{
				GateProj:        sharedGate,
				UpProj:          sharedUp,
				DownProj:        makeLinear(layerPrefix + ".mlp.shared_expert.down_proj"),
				GateUpProj:      fuseDenseGateUp(sharedGate, sharedUp),
				GateUpGateScale: linearGlobalScale(sharedGate),
				GateUpUpScale:   linearGlobalScale(sharedUp),
			}
			if moe.SharedExpert.GateProj == nil || moe.SharedExpert.UpProj == nil || moe.SharedExpert.DownProj == nil {
				return fmt.Errorf("layer %d: missing shared expert weights", i)
			}
			layer.MLP = moe
		} else {
			gate := makeLinear(layerPrefix + ".mlp.gate_proj")
			up := makeLinear(layerPrefix + ".mlp.up_proj")
			mlp := &DenseMLP{
				GateProj:        gate,
				UpProj:          up,
				DownProj:        makeLinear(layerPrefix + ".mlp.down_proj"),
				GateUpProj:      fuseDenseGateUp(gate, up),
				GateUpGateScale: linearGlobalScale(gate),
				GateUpUpScale:   linearGlobalScale(up),
			}
			if mlp.GateProj == nil || mlp.UpProj == nil || mlp.DownProj == nil {
				return fmt.Errorf("layer %d: missing dense mlp projections", i)
			}
			layer.MLP = mlp
		}
		m.Layers[i] = layer
	}
	return nil
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, layer *Layer, cfg *Config) *mlx.Array {
	numHeads := a.NumHeads
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, numHeads, cfg.HeadDim)
	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)

	q = a.QNorm.Forward(q, cfg.RMSNormEps)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	ropeDim, ropeBase, ropeMSScale, ropeFreqs := cfg.FullRopeDim, cfg.FullRopeBase, cfg.FullRopeScale, cfg.FullRopeFreqs
	if layer.IsSliding {
		ropeDim, ropeBase, ropeMSScale, ropeFreqs = cfg.SlidingRopeDim, cfg.SlidingRopeBase, cfg.SlidingRopeScale, nil
	}
	q = nn.ScaleRotaryPart(mlx.RoPEWithFreqs(q, ropeDim, false, ropeBase, 1.0, positions, ropeFreqs), ropeDim, ropeMSScale)
	k = nn.ScaleRotaryPart(mlx.RoPEWithFreqs(k, ropeDim, false, ropeBase, 1.0, positions, ropeFreqs), ropeDim, ropeMSScale)

	var kv nn.SDPAOption
	if c != nil {
		history := c.(cache.Attention).Update(b, k, v)
		kv = nn.WithKVHistory(history)
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
	}
	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(nn.CausalMask()))

	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, numHeads, cfg.HeadDim)
	gate := mlx.ExpandDims(mlx.SoftplusF32(a.GProj.Forward(x)), -1)
	out = mlx.Reshape(mlx.Mul(out, gate), B, L, numHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (m *DenseMLP) Forward(x *mlx.Array, _ *Config) *mlx.Array {
	if m.GateUpProj != nil {
		gateUp := m.GateUpProj.Forward(x)
		gate, up := splitLastDim(gateUp, int32(gateUp.Dim(len(gateUp.Dims())-1))/2)
		gate = applyDenseGlobalScale(gate, m.GateUpGateScale)
		up = applyDenseGlobalScale(up, m.GateUpUpScale)
		return m.DownProj.Forward(mlx.SwiGLU(gate, up))
	}
	return m.DownProj.Forward(mlx.SwiGLU(m.GateProj.Forward(x), m.UpProj.Forward(x)))
}

func (s *SwitchMLP) Forward(x *mlx.Array, indices *mlx.Array, cfg *Config) *mlx.Array {
	if out, ok := s.mappedQuantizedForward(x, indices, cfg); ok {
		return out
	}
	if out, ok := s.mappedDenseForward(x, indices, cfg); ok {
		return out
	}
	// If a mapped path is supported by the model shape but not by the current
	// Metal toolchain, keep the fallback on the unsorted gathered-matmul path.
	// MLX's sorted GatherQMM route can lazily compile Metal tensor-op sources,
	// which are unavailable on macOS 15.
	sortExperts := !s.canMapQuantizedExpert() && !s.canMapDenseExpert()
	return s.gatherForward(x, indices, cfg, sortExperts)
}

func (s *SwitchMLP) canMapQuantizedExpert() bool {
	if !s.UseQuantized {
		return false
	}
	if s.DownWeightQ == nil || s.DownScales == nil || s.DownBiases != nil {
		return false
	}
	if s.UseFusedGateUp {
		if s.GateUpWeightQ == nil || s.GateUpScales == nil || s.GateUpBiases != nil {
			return false
		}
		return mlx.SupportsMoEGatherQMMBlockMapped(s.GateUpGroupSize, s.GateUpBits, s.GateUpMode) &&
			mlx.SupportsMoEGatherQMMBlockMapped(s.DownGroupSize, s.DownBits, s.DownMode)
	}
	if s.GateWeightQ == nil || s.GateScales == nil ||
		s.UpWeightQ == nil || s.UpScales == nil ||
		s.GateBiases != nil || s.UpBiases != nil {
		return false
	}
	return mlx.SupportsMoEGatherQMMBlockMapped(s.GateGroupSize, s.GateBits, s.GateMode) &&
		mlx.SupportsMoEGatherQMMBlockMapped(s.UpGroupSize, s.UpBits, s.UpMode) &&
		mlx.SupportsMoEGatherQMMBlockMapped(s.DownGroupSize, s.DownBits, s.DownMode)
}

func (s *SwitchMLP) canMapDenseExpert() bool {
	if s.UseQuantized || s.DownWeight == nil {
		return false
	}
	if !mlx.SupportsMoEGatherMMBlockMapped(s.DownWeight.DType()) {
		return false
	}
	if s.UseFusedGateUp {
		return s.GateUpWeight != nil &&
			s.GateUpWeight.DType() == s.DownWeight.DType() &&
			mlx.SupportsMoEGatherMMBlockMapped(s.GateUpWeight.DType())
	}
	return s.GateWeight != nil && s.UpWeight != nil &&
		s.GateWeight.DType() == s.DownWeight.DType() &&
		s.UpWeight.DType() == s.DownWeight.DType() &&
		mlx.SupportsMoEGatherMMBlockMapped(s.GateWeight.DType()) &&
		mlx.SupportsMoEGatherMMBlockMapped(s.UpWeight.DType())
}

func (s *SwitchMLP) weightForGatherMM(w *mlx.Array) *mlx.Array {
	if s.DenseWeightsSourceLayout {
		return transposeExpertWeightViewForGatherMM(w)
	}
	return w
}

func (s *SwitchMLP) mappedQuantizedForward(x *mlx.Array, indices *mlx.Array, cfg *Config) (*mlx.Array, bool) {
	if !s.canMapQuantizedExpert() {
		return nil, false
	}

	dims := x.Dims()
	if len(dims) != 3 || dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0 || cfg.NumExpertsPerTok <= 0 {
		return nil, false
	}
	B, L := int32(dims[0]), int32(dims[1])
	tokens := B * L
	if !shouldMapMoEExperts(tokens) {
		return nil, false
	}

	topK := int(cfg.NumExpertsPerTok)
	idxFlat := mlx.Reshape(indices, tokens, cfg.NumExpertsPerTok)
	var experts int
	if s.UseFusedGateUp {
		experts = s.GateUpWeightQ.Dim(0)
	} else {
		experts = s.GateWeightQ.Dim(0)
	}
	expertMap, ok := mlx.NewMoEGatherQMMMap(idxFlat, experts)
	if !ok {
		return nil, false
	}

	xFlat := mlx.Reshape(x, tokens, 1, cfg.HiddenSize)
	if s.UseFusedGateUp {
		gateUp, ok := mlx.FastMoEGatherQMMBlockMapped(xFlat, s.GateUpWeightQ, s.GateUpScales, expertMap, topK, s.GateUpGroupSize, s.GateUpBits, s.GateUpMode)
		if !ok {
			return nil, false
		}
		guDims := gateUp.Dims()
		if len(guDims) != 3 || guDims[2]%2 != 0 {
			return nil, false
		}
		mid := int32(guDims[len(guDims)-1] / 2)
		gate := mlx.SliceStartStop(gateUp, []int32{0, 0, 0}, []int32{tokens, cfg.NumExpertsPerTok, mid})
		up := mlx.SliceStartStop(gateUp, []int32{0, 0, mid}, []int32{tokens, cfg.NumExpertsPerTok, int32(guDims[len(guDims)-1])})
		hidden := mlx.SwiGLU(gate, up)
		down, ok := mlx.FastMoEGatherQMMBlockMapped(hidden, s.DownWeightQ, s.DownScales, expertMap, topK, s.DownGroupSize, s.DownBits, s.DownMode)
		if !ok {
			return nil, false
		}
		return mlx.Reshape(down, B, L, cfg.NumExpertsPerTok, cfg.HiddenSize), true
	}

	gate, ok := mlx.FastMoEGatherQMMBlockMapped(xFlat, s.GateWeightQ, s.GateScales, expertMap, topK, s.GateGroupSize, s.GateBits, s.GateMode)
	if !ok {
		return nil, false
	}
	up, ok := mlx.FastMoEGatherQMMBlockMapped(xFlat, s.UpWeightQ, s.UpScales, expertMap, topK, s.UpGroupSize, s.UpBits, s.UpMode)
	if !ok {
		return nil, false
	}

	gate = mlx.Reshape(gate, tokens, cfg.NumExpertsPerTok, 1, int32(gate.Dim(2)))
	up = mlx.Reshape(up, tokens, cfg.NumExpertsPerTok, 1, int32(up.Dim(2)))
	var hidden *mlx.Array
	if s.GateGlobalScale != nil {
		if fused, ok := mlx.FastSwiGLUGatheredGateScale(gate, up, s.GateGlobalScale, idxFlat); ok {
			hidden = fused
		}
	}
	if hidden == nil {
		gate = applyExpertGlobalScale(gate, s.GateGlobalScale, idxFlat)
		hidden = mlx.SwiGLU(gate, up)
	}
	hidden = mlx.Reshape(hidden, tokens, cfg.NumExpertsPerTok, int32(hidden.Dim(3)))

	down, ok := mlx.FastMoEGatherQMMBlockMapped(hidden, s.DownWeightQ, s.DownScales, expertMap, topK, s.DownGroupSize, s.DownBits, s.DownMode)
	if !ok {
		return nil, false
	}
	return mlx.Reshape(down, B, L, cfg.NumExpertsPerTok, cfg.HiddenSize), true
}

func (s *SwitchMLP) mappedDenseForward(x *mlx.Array, indices *mlx.Array, cfg *Config) (*mlx.Array, bool) {
	if !s.canMapDenseExpert() {
		return nil, false
	}

	dims := x.Dims()
	if len(dims) != 3 || dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0 || cfg.NumExpertsPerTok <= 0 {
		return nil, false
	}
	B, L := int32(dims[0]), int32(dims[1])
	tokens := B * L
	if !shouldMapDenseMoEExperts(tokens) {
		return nil, false
	}

	topK := int(cfg.NumExpertsPerTok)
	idxFlat := mlx.Reshape(indices, tokens, cfg.NumExpertsPerTok)
	var experts int
	if s.UseFusedGateUp {
		experts = s.GateUpWeight.Dim(0)
	} else {
		experts = s.GateWeight.Dim(0)
	}
	expertMap, ok := mlx.NewMoEExpertMap(idxFlat, experts)
	if !ok {
		return nil, false
	}

	xFlat := mlx.Reshape(x, tokens, 1, cfg.HiddenSize)
	if s.UseFusedGateUp {
		gateUp, ok := mlx.FastMoEGatherMMBlockMapped(xFlat, s.GateUpWeight, expertMap, topK)
		if !ok {
			return nil, false
		}
		guDims := gateUp.Dims()
		if len(guDims) != 3 || guDims[2]%2 != 0 {
			return nil, false
		}
		mid := int32(guDims[len(guDims)-1] / 2)
		gate := mlx.SliceStartStop(gateUp, []int32{0, 0, 0}, []int32{tokens, cfg.NumExpertsPerTok, mid})
		up := mlx.SliceStartStop(gateUp, []int32{0, 0, mid}, []int32{tokens, cfg.NumExpertsPerTok, int32(guDims[len(guDims)-1])})
		hidden := mlx.SwiGLU(gate, up)
		down, ok := mlx.FastMoEGatherMMBlockMapped(hidden, s.DownWeight, expertMap, topK)
		if !ok {
			return nil, false
		}
		return mlx.Reshape(down, B, L, cfg.NumExpertsPerTok, cfg.HiddenSize), true
	}

	gate, ok := mlx.FastMoEGatherMMBlockMapped(xFlat, s.GateWeight, expertMap, topK)
	if !ok {
		return nil, false
	}
	up, ok := mlx.FastMoEGatherMMBlockMapped(xFlat, s.UpWeight, expertMap, topK)
	if !ok {
		return nil, false
	}

	gate = mlx.Reshape(gate, tokens, cfg.NumExpertsPerTok, 1, int32(gate.Dim(2)))
	up = mlx.Reshape(up, tokens, cfg.NumExpertsPerTok, 1, int32(up.Dim(2)))
	var hidden *mlx.Array
	if s.GateGlobalScale != nil {
		if fused, ok := mlx.FastSwiGLUGatheredGateScale(gate, up, s.GateGlobalScale, idxFlat); ok {
			hidden = fused
		}
	}
	if hidden == nil {
		gate = applyExpertGlobalScale(gate, s.GateGlobalScale, idxFlat)
		hidden = mlx.SwiGLU(gate, up)
	}
	hidden = mlx.Reshape(hidden, tokens, cfg.NumExpertsPerTok, int32(hidden.Dim(3)))

	down, ok := mlx.FastMoEGatherMMBlockMapped(hidden, s.DownWeight, expertMap, topK)
	if !ok {
		return nil, false
	}
	return mlx.Reshape(down, B, L, cfg.NumExpertsPerTok, cfg.HiddenSize), true
}

func (s *SwitchMLP) gatherForward(x *mlx.Array, indices *mlx.Array, cfg *Config, sortExperts bool) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := cfg.NumExpertsPerTok

	xExpanded := mlx.ExpandDims(mlx.ExpandDims(x, -2), -2)
	xFlat := mlx.Reshape(xExpanded, B*L, 1, 1, cfg.HiddenSize)
	idxFlat := mlx.Reshape(indices, B*L, topK)
	doSort := sortExperts && shouldMapMoEExperts(B*L)
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
		if s.UseFusedGateUp {
			gateUp := mlx.GatherQMM(xFlat, s.GateUpWeightQ, s.GateUpScales, s.GateUpBiases, nil, idxFlat, true, s.GateUpGroupSize, s.GateUpBits, s.GateUpMode, doSort)
			guDims := gateUp.Dims()
			mid := int32(guDims[len(guDims)-1] / 2)
			gate = mlx.SliceStartStop(gateUp, []int32{0, 0, 0, 0}, []int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), mid})
			up = mlx.SliceStartStop(gateUp, []int32{0, 0, 0, mid}, []int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), int32(guDims[len(guDims)-1])})
			hidden = mlx.SwiGLU(gate, up)
		} else {
			gate = mlx.GatherQMM(xFlat, s.GateWeightQ, s.GateScales, s.GateBiases, nil, idxFlat, true, s.GateGroupSize, s.GateBits, s.GateMode, doSort)
			up = mlx.GatherQMM(xFlat, s.UpWeightQ, s.UpScales, s.UpBiases, nil, idxFlat, true, s.UpGroupSize, s.UpBits, s.UpMode, doSort)
			if s.GateGlobalScale != nil {
				if fused, ok := mlx.FastSwiGLUGatheredGateScale(gate, up, s.GateGlobalScale, idxFlat); ok {
					hidden = fused
				}
			}
			if hidden == nil {
				gate = applyExpertGlobalScale(gate, s.GateGlobalScale, idxFlat)
				hidden = mlx.SwiGLU(gate, up)
			}
		}
		down = mlx.GatherQMM(hidden, s.DownWeightQ, s.DownScales, s.DownBiases, nil, idxFlat, true, s.DownGroupSize, s.DownBits, s.DownMode, doSort)
	} else {
		if s.UseFusedGateUp && s.GateUpWeight != nil {
			gateUp := mlx.GatherMM(xFlat, s.weightForGatherMM(s.GateUpWeight), nil, idxFlat, doSort)
			guDims := gateUp.Dims()
			mid := int32(guDims[len(guDims)-1] / 2)
			gate = mlx.SliceStartStop(gateUp, []int32{0, 0, 0, 0}, []int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), mid})
			up = mlx.SliceStartStop(gateUp, []int32{0, 0, 0, mid}, []int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), int32(guDims[len(guDims)-1])})
			hidden = mlx.SwiGLU(gate, up)
		} else {
			gate = mlx.GatherMM(xFlat, s.weightForGatherMM(s.GateWeight), nil, idxFlat, doSort)
			up = mlx.GatherMM(xFlat, s.weightForGatherMM(s.UpWeight), nil, idxFlat, doSort)
			hidden = mlx.SwiGLU(gate, up)
		}
		down = mlx.GatherMM(hidden, s.weightForGatherMM(s.DownWeight), nil, idxFlat, doSort)
	}
	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}
	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

func scaleScoresByExpert(scores, inds, globalScale *mlx.Array) *mlx.Array {
	if globalScale == nil {
		return scores
	}
	scale := globalScale
	if scale.DType() != scores.DType() {
		scale = scale.AsType(scores.DType())
	}
	if scale.NumDims() == 0 || (scale.NumDims() == 1 && scale.Dim(0) == 1) {
		return mlx.Mul(scores, scale)
	}
	return mlx.Mul(scores, mlx.Take(scale, inds, 0))
}

func (m *SparseMoE) route(xFlat *mlx.Array, cfg *Config) (scores, inds *mlx.Array, scalesFolded bool) {
	gates := m.Gate.Forward(xFlat)
	var probs, neg *mlx.Array
	if m.EScoreCorrectionBias != nil {
		if m.SwitchMLP != nil && m.SwitchMLP.UpGlobalScale != nil && m.SwitchMLP.DownGlobalScale != nil {
			if scores, inds, ok := mlx.FastSigmoidTopKRouterScaled(gates, m.EScoreCorrectionBias, m.SwitchMLP.UpGlobalScale, m.SwitchMLP.DownGlobalScale, int(cfg.NumExpertsPerTok), cfg.NormTopKProb && cfg.NumExpertsPerTok > 1); ok {
				return scores, inds, true
			}
		}
		if scores, inds, ok := mlx.FastSigmoidTopKRouter(gates, m.EScoreCorrectionBias, int(cfg.NumExpertsPerTok), cfg.NormTopKProb && cfg.NumExpertsPerTok > 1); ok {
			return scores, inds, false
		}
		gates = gates.AsType(mlx.DTypeFloat32)
		probs, neg = mlx.SigmoidRouter(gates, m.EScoreCorrectionBias)
	} else {
		gates = gates.AsType(mlx.DTypeFloat32)
		probs = mlx.Sigmoid(gates)
		neg = mlx.Neg(probs)
	}
	inds = mlx.Argpartition(neg, int(cfg.NumExpertsPerTok)-1, -1)
	inds = mlx.SliceStartStop(inds, []int32{0, 0}, []int32{int32(xFlat.Dim(0)), cfg.NumExpertsPerTok})
	scores = mlx.TakeAlongAxis(probs, inds, -1)
	if cfg.NormTopKProb && cfg.NumExpertsPerTok > 1 {
		scores = mlx.Div(scores, mlx.Sum(scores, -1, true))
	}
	return scores, inds, false
}

func (m *SparseMoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	return m.forward(x, nil, cfg)
}

func (m *SparseMoE) ForwardAdd(x, residual *mlx.Array, cfg *Config) *mlx.Array {
	return m.forward(x, residual, cfg)
}

func (m *SparseMoE) forward(x, residual *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	BL := B * L

	shared := m.SharedExpert.Forward(x, cfg)
	xFlat := mlx.Reshape(x, BL, cfg.HiddenSize)
	scores, inds, scalesFolded := m.route(xFlat, cfg)
	if !scalesFolded {
		scores = scaleScoresByExpert(scores, inds, m.SwitchMLP.UpGlobalScale)
		scores = scaleScoresByExpert(scores, inds, m.SwitchMLP.DownGlobalScale)
	}

	expertOut := m.SwitchMLP.Forward(x, inds, cfg)
	scoreDims := mlx.Reshape(scores, B, L, cfg.NumExpertsPerTok)
	if residual != nil {
		return moeWeightedSumAdd2(expertOut, scoreDims, shared, residual, x.DType(), cfg.MoeRoutedScalingFactor)
	}
	return moeWeightedSumAdd(expertOut, scoreDims, shared, x.DType(), cfg.MoeRoutedScalingFactor)
}

func moeWeightedSum(expertOut, scores *mlx.Array, dtype mlx.DType, scale float32) *mlx.Array {
	if y, ok := mlx.FastMoEWeightedSum(expertOut, scores, nil, nil, dtype, scale); ok {
		return y
	}
	y := mlx.Sum(mlx.Mul(expertOut, mlx.ExpandDims(scores.AsType(expertOut.DType()), -1)), 2, false).AsType(dtype)
	if scale != 1 {
		y = mlx.MulScalar(y, scale)
	}
	return y
}

func moeWeightedSumAdd(expertOut, scores, shared *mlx.Array, dtype mlx.DType, scale float32) *mlx.Array {
	if y, ok := mlx.FastMoEWeightedSum(expertOut, scores, shared, nil, dtype, scale); ok {
		return y
	}
	return mlx.Add(moeWeightedSum(expertOut, scores, dtype, scale), shared)
}

func moeWeightedSumAdd2(expertOut, scores, addA, addB *mlx.Array, dtype mlx.DType, scale float32) *mlx.Array {
	if y, ok := mlx.FastMoEWeightedSum(expertOut, scores, addA, addB, dtype, scale); ok {
		return y
	}
	return mlx.Add(moeWeightedSumAdd(expertOut, scores, addA, dtype, scale), addB)
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	xn := l.InputNorm.Forward(x, cfg.RMSNormEps)
	r := l.Attention.Forward(xn, b, c, positions, B, L, l, cfg)
	h := mlx.Add(x, r)
	mn := l.PostAttentionNorm.Forward(h, cfg.RMSNormEps)
	if mlp, ok := l.MLP.(MLPBlockAdd); ok {
		return mlp.ForwardAdd(mn, h, cfg)
	}
	r = l.MLP.Forward(mn, cfg)
	return mlx.Add(h, r)
}

// Laguna prefill is faster in cache-backed 512-token slices than in the
// runner's larger default chunk because attention work grows with query span.
const lagunaPrefillChunkSize = 512

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	// Keep Laguna's long prefill on the smaller attention shape that benchmarks
	// well on Metal without changing the runner's global chunking contract.
	if m.shouldChunkPrefill(b, caches, B, L) {
		return m.forwardChunked(b, caches, int(L))
	}

	return m.forward(b, caches, B, L)
}

func (m *Model) shouldChunkPrefill(b *batch.Batch, caches []cache.Cache, B, L int32) bool {
	if B != 1 || L <= lagunaPrefillChunkSize {
		return false
	}
	if len(b.SeqOffsets) != 1 || len(b.SeqQueryLens) != 1 || b.SeqQueryLens[0] != L {
		return false
	}
	if len(caches) < len(m.Layers) {
		return false
	}
	for i := range m.Layers {
		if caches[i] == nil {
			return false
		}
	}
	return true
}

func (m *Model) forwardChunked(b *batch.Batch, caches []cache.Cache, total int) *mlx.Array {
	hidden := make([]*mlx.Array, 0, (total+lagunaPrefillChunkSize-1)/lagunaPrefillChunkSize)
	for start := 0; start < total; start += lagunaPrefillChunkSize {
		stop := min(start+lagunaPrefillChunkSize, total)
		chunkIDs := mlx.SliceStartStop(b.InputIDs, []int32{0, int32(start)}, []int32{1, int32(stop)})
		chunk := &batch.Batch{
			InputIDs:     chunkIDs,
			SeqOffsets:   []int32{b.SeqOffsets[0] + int32(start)},
			SeqQueryLens: []int32{int32(stop - start)},
		}
		hidden = append(hidden, m.forward(chunk, caches, 1, int32(stop-start)))
	}
	return mlx.Concatenate(hidden, 1)
}

func (m *Model) forward(b *batch.Batch, caches []cache.Cache, B, L int32) *mlx.Array {
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	h := m.EmbedTokens.Forward(b.InputIDs)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
	}
	return m.Norm.Forward(h, m.RMSNormEps)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(x)
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
