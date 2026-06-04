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

type DenseMLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
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

func fuseExpertStacks(a, b *mlx.Array, axis int) *mlx.Array {
	if a == nil || !a.Valid() || b == nil || !b.Valid() {
		return nil
	}
	out := mlx.Concatenate([]*mlx.Array{a, b}, axis).Clone()
	mlx.Eval(out)
	return out
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
		s := tensors[key+"_scale"]
		if s == nil {
			s = tensors[key+".scale"]
		}
		if s == nil {
			return &stackedExpertWeights{Weight: w}
		}
		qb := tensors[key+"_qbias"]
		if qb == nil {
			qb = tensors[key+".bias"]
		}
		globalScale, _ := combinedTensorGlobalScale(tensors, key)
		gs, b, m := model.ResolveLinearQuantParams(cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant, key, w, s)
		if useQuantized && supportsGatherQMM(m, b) {
			return &stackedExpertWeights{Weight: w, Scales: s, Biases: qb, GlobalScales: globalScale, Bits: b, GroupSize: gs, Mode: m}
		}
		deq := mlx.Dequantize(w, s, qb, gs, b, m)
		if globalScale != nil {
			deq = mlx.Mul(deq, globalScale)
		}
		return &stackedExpertWeights{Weight: deq, GlobalScales: globalScale, Bits: b, GroupSize: gs, Mode: m}
	}
	return nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	prefix := resolveWeightPrefix(tensors)
	cfg := m.Config
	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)

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
		m.LMHead = m.EmbedTokens.AsLinear()
	} else if lmHead := linears.Make("lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else if lmHead := linears.Make(prefix + "lm_head"); lmHead != nil {
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

		layer.Attention.QProj = linears.Make(layerPrefix + ".self_attn.q_proj")
		layer.Attention.KProj = linears.Make(layerPrefix + ".self_attn.k_proj")
		layer.Attention.VProj = linears.Make(layerPrefix + ".self_attn.v_proj")
		layer.Attention.OProj = linears.Make(layerPrefix + ".self_attn.o_proj")
		layer.Attention.GProj = linears.Make(layerPrefix + ".self_attn.g_proj")
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
			moe := &SparseMoE{Gate: linears.Make(layerPrefix + ".mlp.gate")}
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
				sw.GateWeight = transposeExpertWeightForGatherMM(gateW.Weight)
				sw.UpWeight = transposeExpertWeightForGatherMM(upW.Weight)
				sw.DownWeight = transposeExpertWeightForGatherMM(downW.Weight)
				sw.GateUpWeight = fuseExpertStacks(sw.GateWeight, sw.UpWeight, 2)
				sw.UseFusedGateUp = sw.GateUpWeight != nil
			}
			moe.SwitchMLP = sw
			moe.SharedExpert = &DenseMLP{
				GateProj: linears.Make(layerPrefix + ".mlp.shared_expert.gate_proj"),
				UpProj:   linears.Make(layerPrefix + ".mlp.shared_expert.up_proj"),
				DownProj: linears.Make(layerPrefix + ".mlp.shared_expert.down_proj"),
			}
			if moe.SharedExpert.GateProj == nil || moe.SharedExpert.UpProj == nil || moe.SharedExpert.DownProj == nil {
				return fmt.Errorf("layer %d: missing shared expert weights", i)
			}
			layer.MLP = moe
		} else {
			mlp := &DenseMLP{
				GateProj: linears.Make(layerPrefix + ".mlp.gate_proj"),
				UpProj:   linears.Make(layerPrefix + ".mlp.up_proj"),
				DownProj: linears.Make(layerPrefix + ".mlp.down_proj"),
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
	return m.DownProj.Forward(mlx.SwiGLU(m.GateProj.Forward(x), m.UpProj.Forward(x)))
}

func (s *SwitchMLP) Forward(x *mlx.Array, indices *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := cfg.NumExpertsPerTok

	xExpanded := mlx.ExpandDims(mlx.ExpandDims(x, -2), -2)
	xFlat := mlx.Reshape(xExpanded, B*L, 1, 1, cfg.HiddenSize)
	idxFlat := mlx.Reshape(indices, B*L, topK)
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
		if s.UseFusedGateUp {
			gateUp := mlx.GatherQMM(xFlat, s.GateUpWeightQ, s.GateUpScales, s.GateUpBiases, nil, idxFlat, true, s.GateUpGroupSize, s.GateUpBits, s.GateUpMode, doSort)
			guDims := gateUp.Dims()
			mid := int32(guDims[len(guDims)-1] / 2)
			gate = mlx.SliceStartStop(gateUp, []int32{0, 0, 0, 0}, []int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), mid})
			up = mlx.SliceStartStop(gateUp, []int32{0, 0, 0, mid}, []int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), int32(guDims[len(guDims)-1])})
			hidden = mlx.SwiGLU(gate, up)
		} else {
			gate = mlx.GatherQMM(xFlat, s.GateWeightQ, s.GateScales, s.GateBiases, nil, idxFlat, true, s.GateGroupSize, s.GateBits, s.GateMode, doSort)
			if s.GateGlobalScale != nil {
				gate = mlx.Mul(gate, mlx.Take(s.GateGlobalScale, idxFlat, 0))
			}
			up = mlx.GatherQMM(xFlat, s.UpWeightQ, s.UpScales, s.UpBiases, nil, idxFlat, true, s.UpGroupSize, s.UpBits, s.UpMode, doSort)
			if s.UpGlobalScale != nil {
				up = mlx.Mul(up, mlx.Take(s.UpGlobalScale, idxFlat, 0))
			}
			hidden = mlx.SwiGLU(gate, up)
		}
		down = mlx.GatherQMM(hidden, s.DownWeightQ, s.DownScales, s.DownBiases, nil, idxFlat, true, s.DownGroupSize, s.DownBits, s.DownMode, doSort)
		if s.DownGlobalScale != nil {
			down = mlx.Mul(down, mlx.Take(s.DownGlobalScale, idxFlat, 0))
		}
	} else {
		if s.UseFusedGateUp && s.GateUpWeight != nil {
			gateUp := mlx.GatherMM(xFlat, s.GateUpWeight, nil, idxFlat, doSort)
			guDims := gateUp.Dims()
			mid := int32(guDims[len(guDims)-1] / 2)
			gate = mlx.SliceStartStop(gateUp, []int32{0, 0, 0, 0}, []int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), mid})
			up = mlx.SliceStartStop(gateUp, []int32{0, 0, 0, mid}, []int32{int32(guDims[0]), int32(guDims[1]), int32(guDims[2]), int32(guDims[len(guDims)-1])})
			hidden = mlx.SwiGLU(gate, up)
		} else {
			gate = mlx.GatherMM(xFlat, s.GateWeight, nil, idxFlat, doSort)
			up = mlx.GatherMM(xFlat, s.UpWeight, nil, idxFlat, doSort)
			hidden = mlx.SwiGLU(gate, up)
		}
		down = mlx.GatherMM(hidden, s.DownWeight, nil, idxFlat, doSort)
	}
	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}
	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

func (m *SparseMoE) route(xFlat *mlx.Array, cfg *Config) (scores, inds *mlx.Array) {
	gates := m.Gate.Forward(xFlat).AsType(mlx.DTypeFloat32)
	var probs, neg *mlx.Array
	if m.EScoreCorrectionBias != nil {
		probs, neg = mlx.SigmoidRouter(gates, m.EScoreCorrectionBias)
	} else {
		probs = mlx.Sigmoid(gates)
		neg = mlx.Neg(probs)
	}
	inds = mlx.Argpartition(neg, int(cfg.NumExpertsPerTok)-1, -1)
	inds = mlx.SliceStartStop(inds, []int32{0, 0}, []int32{int32(xFlat.Dim(0)), cfg.NumExpertsPerTok})
	scores = mlx.TakeAlongAxis(probs, inds, -1)
	if cfg.NormTopKProb && cfg.NumExpertsPerTok > 1 {
		scores = mlx.Div(scores, mlx.Sum(scores, -1, true))
	}
	return scores, inds
}

func (m *SparseMoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	BL := B * L

	shared := m.SharedExpert.Forward(x, cfg)
	xFlat := mlx.Reshape(x, BL, cfg.HiddenSize)
	scores, inds := m.route(xFlat, cfg)
	scores = scores.AsType(x.DType())

	expertOut := m.SwitchMLP.Forward(x, inds, cfg)
	routed := mlx.Sum(mlx.Mul(expertOut, mlx.ExpandDims(mlx.Reshape(scores, B, L, cfg.NumExpertsPerTok), -1)), 2, false)
	if cfg.MoeRoutedScalingFactor != 1 {
		routed = mlx.MulScalar(routed, cfg.MoeRoutedScalingFactor)
	}
	return mlx.Add(routed, shared)
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	r := l.Attention.Forward(l.InputNorm.Forward(x, cfg.RMSNormEps), b, c, positions, B, L, l, cfg)
	h := mlx.Add(x, r)
	r = l.MLP.Forward(l.PostAttentionNorm.Forward(h, cfg.RMSNormEps), cfg)
	return mlx.Add(h, r)
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
