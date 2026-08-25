// Package nemotron_h provides the Nemotron-H text model implementation for MLX.
package nemotron_h

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
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
	base.Register("NemotronH_Nano_VL_V2", newModel)
	base.Register("NemotronH_Nano_Omni_Reasoning_V3", newModel)
	base.Register("NemotronHForCausalLM", newModel)
}

var (
	_ base.Model      = (*Model)(nil)
	_ base.SelfDraft  = (*Model)(nil)
	_ base.DraftModel = (*mtpDraft)(nil)
)

type Config struct {
	ModelType             string  `json:"model_type"`
	VocabSize             int32   `json:"vocab_size"`
	HiddenSize            int32   `json:"hidden_size"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	HybridOverridePattern string  `json:"hybrid_override_pattern"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	HeadDim               int32   `json:"head_dim"`
	LayerNormEpsilon      float32 `json:"layer_norm_epsilon"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	AttentionBias         bool    `json:"attention_bias"`
	MLPBias               bool    `json:"mlp_bias"`
	UseBias               bool    `json:"use_bias"`

	ConvKernel     int32 `json:"conv_kernel"`
	SSMStateSize   int32 `json:"ssm_state_size"`
	MambaNumHeads  int32 `json:"mamba_num_heads"`
	MambaHeadDim   int32 `json:"mamba_head_dim"`
	NGroups        int32 `json:"n_groups"`
	MambaChunkSize int32 `json:"chunk_size"`

	NRoutedExperts                  int32   `json:"n_routed_experts"`
	NSharedExperts                  int32   `json:"n_shared_experts"`
	NumExpertsPerTok                int32   `json:"num_experts_per_tok"`
	MoEIntermediateSize             int32   `json:"moe_intermediate_size"`
	MoESharedExpertIntermediateSize int32   `json:"moe_shared_expert_intermediate_size"`
	RoutedScalingFactor             float32 `json:"routed_scaling_factor"`
	ExpertGroupCount                int32   `json:"n_group"`
	ExpertGroupUsedCount            int32   `json:"topk_group"`
	NormTopKProb                    bool    `json:"norm_topk_prob"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	LayerTypes []byte  `json:"-"`
	AttnScale  float32 `json:"-"`
}

type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	MTP *MTPHead

	tok *tokenizer.Tokenizer
	*Config

	weightPrefix string
}

type MTPHead struct {
	Enorm  *nn.RMSNorm
	Hnorm  *nn.RMSNorm
	FC     nn.LinearLayer
	Layers []*Layer
	Norm   *nn.RMSNorm
}

type Layer struct {
	Norm *nn.RMSNorm

	Type      byte
	Mamba     *Mamba2
	Attention *Attention
	Dense     *DenseMLP
	MoE       *SparseMoE
}

type Mamba2 struct {
	InProj  nn.LinearLayer
	OutProj nn.LinearLayer

	Conv1D     *nn.Conv1d
	DtBias     *mlx.Array
	A          *mlx.Array
	D          *mlx.Array
	NormWeight *mlx.Array
}

type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
}

type DenseMLP struct {
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

type SparseMoE struct {
	Router         nn.LinearLayer
	RouterWeight   *mlx.Array
	CorrectionBias *mlx.Array

	UseQuantized bool

	UpWeight      *mlx.Array
	UpWeightQ     *mlx.Array
	UpScales      *mlx.Array
	UpBiases      *mlx.Array
	UpGroupSize   int
	UpBits        int
	UpMode        string
	DownWeight    *mlx.Array
	DownWeightQ   *mlx.Array
	DownScales    *mlx.Array
	DownBiases    *mlx.Array
	DownGroupSize int
	DownBits      int
	DownMode      string

	SharedUp   nn.LinearLayer
	SharedDown nn.LinearLayer

	FoldedSharedExperts int32
}

type stackedExpertWeights struct {
	Weight    *mlx.Array
	Scales    *mlx.Array
	Biases    *mlx.Array
	Bits      int
	GroupSize int
	Mode      string
}

type configEnvelope struct {
	LLMConfig *Config `json:"llm_config"`
}

func parseConfig(data []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}

	var env configEnvelope
	if err := json.Unmarshal(data, &env); err != nil {
		return Config{}, fmt.Errorf("parse config envelope: %w", err)
	}
	if env.LLMConfig != nil {
		cfg = *env.LLMConfig
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
	if cfg.LayerNormEpsilon == 0 {
		cfg.LayerNormEpsilon = cfg.RMSNormEps
	}
	if cfg.LayerNormEpsilon == 0 {
		cfg.LayerNormEpsilon = 1e-5
	}
	if cfg.MaxPositionEmbeddings <= 0 {
		cfg.MaxPositionEmbeddings = 4096
	}
	if cfg.ConvKernel <= 0 {
		cfg.ConvKernel = 4
	}
	if cfg.SSMStateSize <= 0 {
		cfg.SSMStateSize = 128
	}
	if cfg.MambaNumHeads <= 0 {
		cfg.MambaNumHeads = 128
	}
	if cfg.MambaHeadDim <= 0 {
		cfg.MambaHeadDim = 64
	}
	if cfg.NGroups <= 0 {
		cfg.NGroups = 1
	}
	if cfg.MambaChunkSize <= 0 {
		cfg.MambaChunkSize = 128
	}
	if cfg.RoutedScalingFactor == 0 {
		cfg.RoutedScalingFactor = 1
	}
	if cfg.ExpertGroupCount <= 0 {
		cfg.ExpertGroupCount = 1
	}
	if cfg.ExpertGroupUsedCount <= 0 {
		cfg.ExpertGroupUsedCount = cfg.ExpertGroupCount
	}
	if cfg.NRoutedExperts > 0 && cfg.ExpertGroupCount > 0 && cfg.NRoutedExperts%cfg.ExpertGroupCount != 0 {
		return Config{}, fmt.Errorf("n_routed_experts %d must be divisible by n_group %d", cfg.NRoutedExperts, cfg.ExpertGroupCount)
	}

	pattern := strings.TrimSpace(cfg.HybridOverridePattern)
	if pattern == "" {
		return Config{}, fmt.Errorf("hybrid_override_pattern must be set")
	}
	if len(pattern) != int(cfg.NumHiddenLayers) {
		return Config{}, fmt.Errorf("hybrid_override_pattern length %d does not match num_hidden_layers %d", len(pattern), cfg.NumHiddenLayers)
	}
	cfg.LayerTypes = []byte(pattern)
	for i, t := range cfg.LayerTypes {
		switch t {
		case 'M', '*', 'A', 'E', '-':
		default:
			return Config{}, fmt.Errorf("unsupported layer type %q at layer %d", t, i)
		}
	}

	cfg.AttnScale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	return cfg, nil
}

func newModel(root *model.Root) (base.Model, error) {
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
	for i, typ := range cfg.LayerTypes {
		m.Layers[i] = &Layer{Type: typ}
	}
	return m, nil
}

type tensorPathLayout struct {
	containerPrefix string
	backbonePrefix  string
}

func (l tensorPathLayout) path(suffix string) string {
	return l.containerPrefix + l.backbonePrefix + suffix
}

func resolveTensorPathLayout(tensors map[string]*mlx.Array) tensorPathLayout {
	for _, layout := range []tensorPathLayout{
		{containerPrefix: "language_model.", backbonePrefix: "backbone."},
		{containerPrefix: "", backbonePrefix: "backbone."},
		{containerPrefix: "model.language_model.", backbonePrefix: "backbone."},
		{containerPrefix: "language_model.", backbonePrefix: "model.backbone."},
		{containerPrefix: "", backbonePrefix: "model.backbone."},
	} {
		if tensors[layout.path("embeddings.weight")] != nil {
			return layout
		}
	}
	return tensorPathLayout{containerPrefix: "language_model.", backbonePrefix: "backbone."}
}

func sanitizeConvWeight(w *mlx.Array) *mlx.Array {
	if w == nil {
		return nil
	}
	if w.NumDims() == 3 {
		if w.Dim(1) == 1 {
			return mlx.Reshape(w, int32(w.Dim(0)), int32(w.Dim(2)))
		}
		if w.Dim(2) == 1 {
			return mlx.Reshape(w, int32(w.Dim(0)), int32(w.Dim(1)))
		}
	}
	return w
}

func stackAndClone(parts []*mlx.Array) *mlx.Array {
	if len(parts) == 0 {
		return nil
	}
	return mlx.Stack(parts, 0).Clone()
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
	for _, key := range keys {
		if key == "" {
			continue
		}
		if tensors[key] != nil {
			delete(tensors, key)
		}
	}
}

func combinedTensorGlobalScale(tensors map[string]*mlx.Array, key string) (*mlx.Array, []string) {
	var keys []string
	weightGlobal := tensors[key+".global_scale"]
	if weightGlobal == nil {
		weightGlobal = tensors[key+".weight.global_scale"]
	}
	if weightGlobal != nil {
		keys = append(keys, key+".global_scale", key+".weight.global_scale")
	}
	if tensors[key+".input_global_scale"] != nil || tensors[key+".weight.input_global_scale"] != nil {
		keys = append(keys, key+".input_global_scale", key+".weight.input_global_scale")
	}
	return weightGlobal, keys
}

func applyExpertWeightGlobalScale(weight, scale *mlx.Array) *mlx.Array {
	if scale == nil {
		return weight
	}
	if scale.DType() != weight.DType() {
		scale = scale.AsType(weight.DType())
	}
	switch scale.NumDims() {
	case 1:
		if scale.Dim(0) == 1 {
			break
		}
		if weight.NumDims() == 3 && scale.Dim(0) == weight.Dim(0) {
			scale = mlx.Reshape(scale, int32(scale.Dim(0)), 1, 1)
		} else if weight.NumDims() == 2 && scale.Dim(0) == weight.Dim(0) {
			scale = mlx.Reshape(scale, int32(scale.Dim(0)), 1)
		}
	case 2:
		if weight.NumDims() == 3 && scale.Dim(0) == weight.Dim(0) && scale.Dim(1) == weight.Dim(1) {
			scale = mlx.ExpandDims(scale, -1)
		}
	}
	return mlx.Mul(weight, scale)
}

func transposeExpertWeightForGatherMM(w *mlx.Array) *mlx.Array {
	if w == nil || !w.Valid() || w.NumDims() != 3 {
		return w
	}
	return mlx.Transpose(w, 0, 2, 1).Clone()
}

func sliceAxis(a *mlx.Array, axis int, start, stop int32) *mlx.Array {
	dims := a.Dims()
	starts := make([]int32, len(dims))
	stops := make([]int32, len(dims))
	for i, d := range dims {
		stops[i] = int32(d)
	}
	starts[axis] = start
	stops[axis] = stop
	return mlx.SliceStartStop(a, starts, stops)
}

func appendAndClone(dst, src *mlx.Array) *mlx.Array {
	return mlx.Concatenate([]*mlx.Array{dst, src}, 0).Clone()
}

func stackSlicesAndClone(slices []*mlx.Array) *mlx.Array {
	if len(slices) == 0 {
		return nil
	}
	return mlx.Stack(slices, 0).Clone()
}

func foldSharedExperts(m *SparseMoE, cfg *Config) bool {
	if m == nil || cfg == nil || cfg.NSharedExperts != 1 {
		return false
	}
	if cfg.MoEIntermediateSize <= 0 || cfg.MoESharedExpertIntermediateSize <= 0 {
		return false
	}
	if cfg.MoESharedExpertIntermediateSize%cfg.MoEIntermediateSize != 0 {
		return false
	}
	parts := cfg.MoESharedExpertIntermediateSize / cfg.MoEIntermediateSize
	if parts <= 0 {
		return false
	}

	if !m.UseQuantized || !foldQuantizedSharedExperts(m, cfg, parts) {
		return false
	}

	m.FoldedSharedExperts = parts
	m.SharedUp = nil
	m.SharedDown = nil
	return true
}

func foldQuantizedSharedExperts(m *SparseMoE, cfg *Config, parts int32) bool {
	up, ok := m.SharedUp.(*nn.QuantizedLinear)
	if !ok || up == nil {
		return false
	}
	down, ok := m.SharedDown.(*nn.QuantizedLinear)
	if !ok || down == nil {
		return false
	}
	if up.Bias != nil || up.GlobalScale != nil || down.Bias != nil || down.GlobalScale != nil {
		return false
	}
	if m.UpWeightQ == nil || m.UpScales == nil || m.DownWeightQ == nil || m.DownScales == nil {
		return false
	}
	if up.GroupSize != m.UpGroupSize || up.Bits != m.UpBits || up.Mode != m.UpMode {
		return false
	}
	if down.GroupSize != m.DownGroupSize || down.Bits != m.DownBits || down.Mode != m.DownMode {
		return false
	}
	if (up.QBiases == nil) != (m.UpBiases == nil) || (down.QBiases == nil) != (m.DownBiases == nil) {
		return false
	}
	if up.Weight.Dim(0) != int(cfg.MoESharedExpertIntermediateSize) || down.Weight.Dim(0) != int(cfg.HiddenSize) {
		return false
	}

	upWeightStack := stackQuantizedUpParts(up.Weight, cfg.MoEIntermediateSize, parts)
	upScaleStack := stackQuantizedUpParts(up.Scales, cfg.MoEIntermediateSize, parts)
	if upWeightStack == nil || upScaleStack == nil {
		return false
	}
	if upWeightStack.Dim(1) != m.UpWeightQ.Dim(1) || upWeightStack.Dim(2) != m.UpWeightQ.Dim(2) {
		return false
	}
	if upScaleStack.Dim(1) != m.UpScales.Dim(1) || upScaleStack.Dim(2) != m.UpScales.Dim(2) {
		return false
	}

	downWeightStack := stackQuantizedDownParts(down.Weight, m.DownWeightQ.Dim(2), parts)
	downScaleStack := stackQuantizedDownParts(down.Scales, m.DownScales.Dim(2), parts)
	if downWeightStack == nil || downScaleStack == nil {
		return false
	}
	if downWeightStack.Dim(1) != m.DownWeightQ.Dim(1) || downWeightStack.Dim(2) != m.DownWeightQ.Dim(2) {
		return false
	}
	if downScaleStack.Dim(1) != m.DownScales.Dim(1) || downScaleStack.Dim(2) != m.DownScales.Dim(2) {
		return false
	}

	var upBiasStack, downBiasStack *mlx.Array
	if m.UpBiases != nil {
		upBiasStack = stackQuantizedUpParts(up.QBiases, cfg.MoEIntermediateSize, parts)
		if upBiasStack == nil || upBiasStack.Dim(1) != m.UpBiases.Dim(1) || upBiasStack.Dim(2) != m.UpBiases.Dim(2) {
			return false
		}
	}
	if m.DownBiases != nil {
		downBiasStack = stackQuantizedDownParts(down.QBiases, m.DownBiases.Dim(2), parts)
		if downBiasStack == nil || downBiasStack.Dim(1) != m.DownBiases.Dim(1) || downBiasStack.Dim(2) != m.DownBiases.Dim(2) {
			return false
		}
	}

	m.UpWeightQ = appendAndClone(m.UpWeightQ, upWeightStack)
	m.UpScales = appendAndClone(m.UpScales, upScaleStack)
	m.DownWeightQ = appendAndClone(m.DownWeightQ, downWeightStack)
	m.DownScales = appendAndClone(m.DownScales, downScaleStack)
	if upBiasStack != nil {
		m.UpBiases = appendAndClone(m.UpBiases, upBiasStack)
	}
	if downBiasStack != nil {
		m.DownBiases = appendAndClone(m.DownBiases, downBiasStack)
	}

	return true
}

func stackQuantizedUpParts(a *mlx.Array, partSize int32, parts int32) *mlx.Array {
	if a == nil || a.NumDims() != 2 || a.Dim(0) != int(partSize*parts) {
		return nil
	}
	slices := make([]*mlx.Array, 0, parts)
	for i := range parts {
		start := i * partSize
		slices = append(slices, sliceAxis(a, 0, start, start+partSize))
	}
	return stackSlicesAndClone(slices)
}

func stackQuantizedDownParts(a *mlx.Array, packedPartSize int, parts int32) *mlx.Array {
	if a == nil || a.NumDims() != 2 || packedPartSize <= 0 || a.Dim(1) != packedPartSize*int(parts) {
		return nil
	}
	slices := make([]*mlx.Array, 0, parts)
	for i := range parts {
		start := int(i) * packedPartSize
		slices = append(slices, sliceAxis(a, 1, int32(start), int32(start+packedPartSize)))
	}
	return stackSlicesAndClone(slices)
}

func tensorAny(tensors map[string]*mlx.Array, keys ...string) (*mlx.Array, string) {
	for _, k := range keys {
		if v := tensors[k]; v != nil {
			return v, k
		}
	}
	return nil, ""
}

func tensorByBase(tensors map[string]*mlx.Array, base string) (*mlx.Array, string) {
	return tensorAny(tensors, base+".weight", base)
}

// loadStackedExpertProjection loads a pre-stacked [experts, out, in] expert
// tensor (as produced by ollama create's TransformStackExperts). It returns
// nil if no stacked tensor is found, so the caller can fall back to
// per-expert collection.
func loadStackedExpertProjection(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, layerPrefix, proj string) *stackedExpertWeights {
	base := fmt.Sprintf("%s.mixer.experts.%s", layerPrefix, proj)
	w, key := tensorByBase(tensors, base)
	if w == nil {
		return nil
	}

	scales := tensors[key+"_scale"]
	globalScale, globalScaleKeys := combinedTensorGlobalScale(tensors, key)
	if scales == nil {
		freeTensorKeys(tensors, append([]string{key}, globalScaleKeys...)...)
		return &stackedExpertWeights{Weight: applyExpertWeightGlobalScale(w, globalScale)}
	}

	qbiases := tensors[key+"_qbias"]
	groupSize, bits, mode := model.ResolveLinearQuantParams(
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant,
		key, w, scales,
	)

	freeTensorKeys(tensors, append([]string{key, key + "_scale"}, globalScaleKeys...)...)
	if qbiases != nil {
		freeTensorKeys(tensors, key+"_qbias")
	}

	if useQuantized && supportsGatherQMM(mode, bits) && globalScale == nil {
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
		Weight:    applyExpertWeightGlobalScale(mlx.Dequantize(w, scales, qbiases, groupSize, bits, mode, nil), globalScale),
		Bits:      bits,
		GroupSize: groupSize,
		Mode:      mode,
	}
}

func collectExpertProjection(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, layerPrefix, proj string, numExperts int32) *stackedExpertWeights {
	weights := make([]*mlx.Array, 0, numExperts)
	scales := make([]*mlx.Array, 0, numExperts)
	biases := make([]*mlx.Array, 0, numExperts)
	consumed := make([]string, 0, numExperts*3)
	bits := 0
	groupSize := 0
	mode := cfg.QuantMode

	for e := range numExperts {
		key := fmt.Sprintf("%s.mixer.experts.%d.%s.weight", layerPrefix, e, proj)
		w := tensors[key]
		if w == nil {
			return nil
		}
		consumed = append(consumed, key)

		scaleKey := key + "_scale"
		scale := tensors[scaleKey]
		if scale == nil {
			weights = append(weights, w)
			continue
		}

		consumed = append(consumed, scaleKey)
		qbiasKey := key + "_qbias"
		qbias := tensors[qbiasKey]
		if qbias != nil {
			consumed = append(consumed, qbiasKey)
		}
		globalScale, globalScaleKeys := combinedTensorGlobalScale(tensors, key)
		consumed = append(consumed, globalScaleKeys...)

		gs, b, m := model.ResolveLinearQuantParams(
			cfg.QuantGroupSize,
			cfg.QuantBits,
			cfg.QuantMode,
			cfg.TensorQuant,
			key,
			w,
			scale,
		)
		if bits == 0 {
			bits = b
			groupSize = gs
			mode = m
		}
		if useQuantized && supportsGatherQMM(m, b) && globalScale == nil {
			weights = append(weights, w)
			scales = append(scales, scale)
			if qbias != nil {
				biases = append(biases, qbias)
			}
			continue
		}

		weights = append(weights, applyExpertWeightGlobalScale(mlx.Dequantize(w, scale, qbias, gs, b, m, nil), globalScale))
	}

	out := &stackedExpertWeights{
		Weight:    stackAndClone(weights),
		Bits:      bits,
		GroupSize: groupSize,
		Mode:      mode,
	}
	if len(scales) == len(weights) {
		out.Scales = stackAndClone(scales)
	}
	if len(biases) == len(weights) {
		out.Biases = stackAndClone(biases)
	}
	freeTensorKeys(tensors, consumed...)
	return out
}

func loadRMSNorm(tensors map[string]*mlx.Array, key string, eps float32) *nn.RMSNorm {
	if weight := tensors[key]; weight != nil {
		return nn.NewRMSNorm(weight, eps)
	}
	return nil
}

func loadRMSNormAny(tensors map[string]*mlx.Array, eps float32, keys ...string) *nn.RMSNorm {
	for _, key := range keys {
		if norm := loadRMSNorm(tensors, key, eps); norm != nil {
			return norm
		}
	}
	return nil
}

func makeLinearAny(linears model.LinearFactory, bases ...string) nn.LinearLayer {
	for _, base := range bases {
		if l := linears.Make(base); l != nil {
			return l
		}
	}
	return nil
}

func hasTensorByBase(tensors map[string]*mlx.Array, base string) bool {
	t, _ := tensorByBase(tensors, base)
	return t != nil
}

func hasMTPHeadTensors(tensors map[string]*mlx.Array) bool {
	if hasTensorByBase(tensors, "mtp.fc") {
		return true
	}
	for name, tensor := range tensors {
		if tensor != nil && strings.HasPrefix(name, "mtp.layers.") && strings.HasSuffix(name, ".eh_proj.weight") {
			return true
		}
	}
	return false
}

func detectMTPLayerType(tensors map[string]*mlx.Array, layerPrefix string) (byte, error) {
	mixerPrefix := layerPrefix + ".mixer"
	switch {
	case hasTensorByBase(tensors, mixerPrefix+".q_proj") ||
		hasTensorByBase(tensors, mixerPrefix+".k_proj") ||
		hasTensorByBase(tensors, mixerPrefix+".v_proj") ||
		hasTensorByBase(tensors, mixerPrefix+".o_proj"):
		return 'A', nil
	case hasTensorByBase(tensors, mixerPrefix+".gate") ||
		hasTensorByBase(tensors, mixerPrefix+".experts.up_proj") ||
		tensors[mixerPrefix+".experts.0.up_proj.weight"] != nil:
		return 'E', nil
	case hasTensorByBase(tensors, mixerPrefix+".up_proj") ||
		hasTensorByBase(tensors, mixerPrefix+".down_proj"):
		return '-', nil
	case hasTensorByBase(tensors, mixerPrefix+".in_proj") ||
		tensors[mixerPrefix+".conv1d.weight"] != nil:
		return 0, fmt.Errorf("recurrent MTP layers are not supported")
	default:
		return 0, fmt.Errorf("missing supported MTP mixer tensors")
	}
}

func mtpLayerPrefixes(tensors map[string]*mlx.Array) ([]string, error) {
	const prefix = "mtp.layers."
	indices := map[int]struct{}{}

	for name, tensor := range tensors {
		if tensor == nil || !strings.HasPrefix(name, prefix) {
			continue
		}
		rest := strings.TrimPrefix(name, prefix)
		indexText, _, ok := strings.Cut(rest, ".")
		if !ok {
			continue
		}
		index, err := strconv.Atoi(indexText)
		if err != nil {
			continue
		}
		indices[index] = struct{}{}
	}

	if len(indices) == 0 {
		return nil, fmt.Errorf("missing MTP layer tensors")
	}
	sorted := make([]int, 0, len(indices))
	for index := range indices {
		sorted = append(sorted, index)
	}
	sort.Ints(sorted)

	prefixes := make([]string, len(sorted))
	for i, index := range sorted {
		prefixes[i] = fmt.Sprintf("mtp.layers.%d", index)
	}
	return prefixes, nil
}

func loadLayer(linears model.LinearFactory, tensors map[string]*mlx.Array, cfg *Config, layerPrefix string, typ byte, useQuantizedExperts bool) (*Layer, error) {
	layer := &Layer{Type: typ}

	norm := loadRMSNorm(tensors, layerPrefix+".norm.weight", cfg.LayerNormEpsilon)
	if norm == nil {
		return nil, fmt.Errorf("missing norm weight")
	}
	layer.Norm = norm

	mixerPrefix := layerPrefix + ".mixer"
	switch layer.Type {
	case 'M':
		mamba := &Mamba2{}
		mamba.InProj = linears.Make(mixerPrefix + ".in_proj")
		mamba.OutProj = linears.Make(mixerPrefix + ".out_proj")
		convWeight := sanitizeConvWeight(tensors[mixerPrefix+".conv1d.weight"])
		convBias := tensors[mixerPrefix+".conv1d.bias"]
		mamba.DtBias = tensors[mixerPrefix+".dt_bias"]
		aLog := tensors[mixerPrefix+".A_log"]
		if aLog != nil {
			mamba.A = mlx.Neg(mlx.Exp(aLog.AsType(mlx.DTypeFloat32)))
		}
		mamba.D = tensors[mixerPrefix+".D"]
		mamba.NormWeight = tensors[mixerPrefix+".norm.weight"]
		if mamba.InProj == nil || mamba.OutProj == nil || convWeight == nil || mamba.DtBias == nil || mamba.A == nil || mamba.D == nil || mamba.NormWeight == nil {
			return nil, fmt.Errorf("missing mamba2 tensors")
		}
		if convWeight.NumDims() != 2 {
			return nil, fmt.Errorf("conv1d weight must be 2D after sanitization, got %dD", convWeight.NumDims())
		}
		// The fused conv kernel needs weight, bias and input to agree on
		// dtype. Promoting matches what the graph conv would do anyway.
		convWeight = convWeight.AsType(mlx.DTypeFloat32)
		if convBias != nil {
			convBias = convBias.AsType(mlx.DTypeFloat32)
		}
		mamba.Conv1D = nn.NewConv1d(mlx.ExpandDims(convWeight, 2), convBias, 1, 0, 1, int32(convWeight.Dim(0)))
		layer.Mamba = mamba
	case '*', 'A':
		attn := &Attention{
			QProj: linears.Make(mixerPrefix + ".q_proj"),
			KProj: linears.Make(mixerPrefix + ".k_proj"),
			VProj: linears.Make(mixerPrefix + ".v_proj"),
			OProj: linears.Make(mixerPrefix + ".o_proj"),
		}
		if attn.QProj == nil || attn.KProj == nil || attn.VProj == nil || attn.OProj == nil {
			return nil, fmt.Errorf("missing attention projections")
		}
		layer.Attention = attn
	case '-':
		dense := &DenseMLP{
			UpProj:   linears.Make(mixerPrefix + ".up_proj"),
			DownProj: linears.Make(mixerPrefix + ".down_proj"),
		}
		if dense.UpProj == nil || dense.DownProj == nil {
			return nil, fmt.Errorf("missing dense mlp projections")
		}
		layer.Dense = dense
	case 'E':
		moe := &SparseMoE{
			Router:         linears.Make(mixerPrefix + ".gate"),
			RouterWeight:   tensors[mixerPrefix+".gate.weight"],
			CorrectionBias: tensors[mixerPrefix+".gate.e_score_correction_bias"],
			SharedUp:       linears.Make(mixerPrefix + ".shared_experts.up_proj"),
			SharedDown:     linears.Make(mixerPrefix + ".shared_experts.down_proj"),
		}
		if moe.Router == nil || moe.RouterWeight == nil || moe.SharedUp == nil || moe.SharedDown == nil {
			return nil, fmt.Errorf("missing moe router or shared expert projections")
		}
		if cfg.NRoutedExperts <= 0 || cfg.NumExpertsPerTok <= 0 {
			return nil, fmt.Errorf("invalid moe config")
		}
		up := loadStackedExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "up_proj")
		if up == nil {
			up = collectExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "up_proj", cfg.NRoutedExperts)
		}
		down := loadStackedExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "down_proj")
		if down == nil {
			down = collectExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "down_proj", cfg.NRoutedExperts)
		}
		if up == nil || down == nil {
			return nil, fmt.Errorf("missing routed expert weights")
		}
		if up.Scales != nil {
			moe.UpWeightQ = up.Weight
			moe.UpScales = up.Scales
			moe.UpBiases = up.Biases
			moe.UpGroupSize = up.GroupSize
			moe.UpBits = up.Bits
			moe.UpMode = up.Mode
		} else {
			moe.UpWeight = transposeExpertWeightForGatherMM(up.Weight)
		}
		if down.Scales != nil {
			moe.DownWeightQ = down.Weight
			moe.DownScales = down.Scales
			moe.DownBiases = down.Biases
			moe.DownGroupSize = down.GroupSize
			moe.DownBits = down.Bits
			moe.DownMode = down.Mode
		} else {
			moe.DownWeight = transposeExpertWeightForGatherMM(down.Weight)
		}
		moe.UseQuantized = moe.UpWeightQ != nil || moe.DownWeightQ != nil
		if moe.UseQuantized {
			foldSharedExperts(moe, cfg)
		}
		layer.MoE = moe
	default:
		return nil, fmt.Errorf("unsupported layer type %q", layer.Type)
	}

	return layer, nil
}

func (m *Model) loadMTPHead(linears model.LinearFactory, tensors map[string]*mlx.Array, useQuantizedExperts bool) error {
	cfg := m.Config
	layerPrefixes, err := mtpLayerPrefixes(tensors)
	if err != nil {
		return fmt.Errorf("mtp head: %w", err)
	}
	firstPrefix := layerPrefixes[0]
	lastPrefix := layerPrefixes[len(layerPrefixes)-1]

	head := &MTPHead{
		Enorm: loadRMSNormAny(tensors, cfg.LayerNormEpsilon,
			"mtp.pre_fc_norm_embedding.weight",
			firstPrefix+".enorm.weight",
		),
		Hnorm: loadRMSNormAny(tensors, cfg.LayerNormEpsilon,
			"mtp.pre_fc_norm_hidden.weight",
			firstPrefix+".hnorm.weight",
		),
		FC: makeLinearAny(linears,
			"mtp.fc",
			firstPrefix+".eh_proj",
		),
		Norm: loadRMSNormAny(tensors, cfg.LayerNormEpsilon,
			"mtp.norm.weight",
			"mtp.shared_head.norm.weight",
			lastPrefix+".final_layernorm.weight",
		),
	}
	if head.Enorm == nil || head.Hnorm == nil || head.FC == nil || head.Norm == nil {
		return fmt.Errorf("mtp head: missing enorm/hnorm/fc/norm tensors")
	}

	head.Layers = make([]*Layer, len(layerPrefixes))
	for i, layerPrefix := range layerPrefixes {
		typ, err := detectMTPLayerType(tensors, layerPrefix)
		if err != nil {
			return fmt.Errorf("mtp layer %s: %w", layerPrefix, err)
		}
		layer, err := loadLayer(linears, tensors, cfg, layerPrefix, typ, useQuantizedExperts)
		if err != nil {
			return fmt.Errorf("mtp layer %s: %w", layerPrefix, err)
		}
		head.Layers[i] = layer
	}
	m.MTP = head
	return nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	layout := resolveTensorPathLayout(tensors)
	m.weightPrefix = layout.containerPrefix
	backbonePrefix := layout.containerPrefix + layout.backbonePrefix
	cfg := m.Config

	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
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

	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, backbonePrefix+"embeddings", cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing embedding weight: %sembeddings.weight", backbonePrefix)
	}

	normWeight := tensors[backbonePrefix+"norm_f.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: %snorm_f.weight", backbonePrefix)
	}
	m.Norm = nn.NewRMSNorm(normWeight, cfg.LayerNormEpsilon)

	if cfg.TieWordEmbeddings {
		m.LMHead = m.EmbedTokens.AsLinear()
	} else if head := linears.Make(layout.containerPrefix + "lm_head"); head != nil {
		m.LMHead = head
	} else if head := linears.Make("lm_head"); head != nil {
		m.LMHead = head
	} else {
		m.LMHead = m.EmbedTokens.AsLinear()
	}

	for i := range cfg.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("%slayers.%d", backbonePrefix, i)
		layer, err := loadLayer(linears, tensors, cfg, layerPrefix, cfg.LayerTypes[i], useQuantizedExperts)
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
		m.Layers[i] = layer
	}
	if hasMTPHeadTensors(tensors) {
		if err := m.loadMTPHead(linears, tensors, useQuantizedExperts); err != nil {
			return err
		}
	}
	return nil
}

func (m *Mamba2) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	dtype := x.DType()
	projected := m.InProj.Forward(x)
	inner := cfg.MambaNumHeads * cfg.MambaHeadDim
	convDim := inner + 2*cfg.NGroups*cfg.SSMStateSize

	gate := mlx.SliceStartStop(projected, []int32{0, 0, 0}, []int32{B, L, inner})
	xBC := mlx.SliceStartStop(projected, []int32{0, 0, inner}, []int32{B, L, inner + convDim}).AsType(mlx.DTypeFloat32)
	dt := mlx.SliceStartStop(projected, []int32{0, 0, inner + convDim}, []int32{B, L, inner + convDim + cfg.MambaNumHeads})

	convTail := cfg.ConvKernel - 1
	var rc *cache.RecurrentCache
	opts := make([]nn.RecurrentOption, 0, 3)
	opts = append(opts, nn.WithConvSiLU())
	if typed, ok := c.(*cache.RecurrentCache); ok {
		rc = typed
		opts = append(opts, nn.WithRecurrentHistory(rc.Get(b, mlx.DTypeFloat32)))
		if splits := rc.SnapshotSplits(int(L)); len(splits) > 0 {
			opts = append(opts, nn.WithSnapshotSplits(splits))
		}
	} else {
		opts = append(opts, nn.WithRecurrentState(
			mlx.Zeros(mlx.DTypeFloat32, int(B), int(convTail), int(convDim)),
			mlx.Zeros(mlx.DTypeFloat32, int(B), int(cfg.MambaNumHeads), int(cfg.MambaHeadDim), int(cfg.SSMStateSize))))
	}

	convOut, convStates := nn.CausalConv1D(b, xBC, m.Conv1D, int(convTail), opts...)
	if dtype != mlx.DTypeFloat32 {
		convOut = convOut.AsType(dtype).AsType(mlx.DTypeFloat32)
	}

	hidden := mlx.SliceStartStop(convOut, []int32{0, 0, 0}, []int32{B, L, inner})
	bState := mlx.SliceStartStop(convOut, []int32{0, 0, inner}, []int32{B, L, inner + cfg.NGroups*cfg.SSMStateSize})
	cState := mlx.SliceStartStop(convOut, []int32{0, 0, inner + cfg.NGroups*cfg.SSMStateSize}, []int32{B, L, inner + 2*cfg.NGroups*cfg.SSMStateSize})

	hidden = mlx.Reshape(hidden, B, L, cfg.MambaNumHeads, cfg.MambaHeadDim)
	bState = mlx.Reshape(bState, B, L, cfg.NGroups, cfg.SSMStateSize)
	cState = mlx.Reshape(cState, B, L, cfg.NGroups, cfg.SSMStateSize)

	y, deltaStates := nn.Mamba2Scan(b, hidden, bState, cState, dt.AsType(mlx.DTypeFloat32),
		mlx.Reshape(m.A.AsType(mlx.DTypeFloat32), cfg.MambaNumHeads),
		mlx.Reshape(m.D.AsType(mlx.DTypeFloat32), cfg.MambaNumHeads),
		mlx.Reshape(m.DtBias.AsType(mlx.DTypeFloat32), cfg.MambaNumHeads),
		opts...)

	y = mlx.Reshape(y, B, L, inner)
	y = gatedGroupRMSNorm(y, gate, m.NormWeight, cfg, dtype)
	out := m.OutProj.Forward(y)
	if rc != nil {
		rc.Put(b, convStates, deltaStates)
	}
	return out
}

// gatedGroupRMSNorm computes RMSNorm(y * SiLU(gate), groups) * weight. The
// weight varies across groups while the norm reduces within one, so it is a
// separate multiply rather than the norm's own weight argument.
func gatedGroupRMSNorm(y, gate, weight *mlx.Array, cfg *Config, dtype mlx.DType) *mlx.Array {
	inner := cfg.MambaNumHeads * cfg.MambaHeadDim
	dims := y.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	groupSize := inner / cfg.NGroups

	y = mlx.Mul(y, mlx.SiLU(gate.AsType(y.DType())))
	y = mlx.RMSNormFn(mlx.Reshape(y, B, L, cfg.NGroups, groupSize), nil, cfg.LayerNormEpsilon)
	y = mlx.Mul(y, mlx.Reshape(weight.AsType(y.DType()), 1, 1, cfg.NGroups, groupSize))
	return mlx.Reshape(y, B, L, inner).AsType(dtype)
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	var kv nn.SDPAOption
	if c != nil {
		history := c.(cache.Attention).Update(b, k, v)
		kv = nn.WithKVHistory(history)
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
	}

	out := nn.ScaledDotProductAttention(b, q, cfg.AttnScale, kv, nn.WithMask(nn.CausalMask()))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	out = a.OProj.Forward(out)
	return out
}

func (m *DenseMLP) Forward(x *mlx.Array) *mlx.Array {
	up := m.UpProj.Forward(x)
	hidden := mlx.ReLUSquared(up)
	out := m.DownProj.Forward(hidden)
	return out
}

func shouldSortMoEExperts(tokens int32) bool {
	return tokens >= 64
}

func (m *SparseMoE) gatherExpertUp(xFlat, idxFlat *mlx.Array, doSort bool) *mlx.Array {
	if m.UpWeightQ != nil {
		return mlx.GatherQMM(
			xFlat,
			m.UpWeightQ,
			m.UpScales,
			m.UpBiases,
			nil,
			idxFlat,
			true,
			m.UpGroupSize,
			m.UpBits,
			m.UpMode,
			doSort,
		)
	}
	return mlx.GatherMM(xFlat, m.UpWeight, nil, idxFlat, doSort)
}

func (m *SparseMoE) gatherExpertDown(hidden, idxFlat *mlx.Array, doSort bool) *mlx.Array {
	if m.DownWeightQ != nil {
		return mlx.GatherQMM(
			hidden,
			m.DownWeightQ,
			m.DownScales,
			m.DownBiases,
			nil,
			idxFlat,
			true,
			m.DownGroupSize,
			m.DownBits,
			m.DownMode,
			doSort,
		)
	}
	return mlx.GatherMM(hidden, m.DownWeight, nil, idxFlat, doSort)
}

func (m *SparseMoE) expertForward(x *mlx.Array, indices *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := int32(indices.Dim(2))

	xExpanded := mlx.ExpandDims(mlx.ExpandDims(x, -2), -2)
	xFlat := mlx.Reshape(xExpanded, B*L, 1, 1, cfg.HiddenSize)
	idxFlat := mlx.Reshape(indices, B*L, topK)

	doSort := shouldSortMoEExperts(B * L)
	var invOrder *mlx.Array
	n := B * L * topK
	if doSort {
		idxAll := mlx.Flatten(idxFlat)
		order := mlx.Argsort(idxAll, 0)
		invOrder = mlx.Argsort(order, 0)
		tokenOrder := mlx.FloorDivideScalar(order, topK)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), tokenOrder, 0), 1)
		idxFlat = mlx.Reshape(mlx.Take(idxAll, order, 0), n, 1)
	}

	up := m.gatherExpertUp(xFlat, idxFlat, doSort)

	hidden := mlx.ReLUSquared(up)
	if hidden.DType() != up.DType() {
		// Keep the activation in the projection dtype so dense and quantized
		// down projections stay on the BF16 fast path.
		hidden = hidden.AsType(up.DType())
	}
	down := m.gatherExpertDown(hidden, idxFlat, doSort)

	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}
	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

func appendFoldedSharedExperts(inds, scores *mlx.Array, cfg *Config, folded int32) (*mlx.Array, *mlx.Array) {
	if folded <= 0 {
		return inds, scores
	}
	dims := inds.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	sharedInds := make([]*mlx.Array, 0, folded+1)
	sharedInds = append(sharedInds, inds)
	for i := range folded {
		idx := mlx.AddScalar(mlx.Zeros(mlx.DTypeInt32, int(B), int(L), 1), float32(cfg.NRoutedExperts+i))
		if idx.DType() != inds.DType() {
			idx = idx.AsType(inds.DType())
		}
		sharedInds = append(sharedInds, idx)
	}
	inds = mlx.Concatenate(sharedInds, 2)

	ones := mlx.AddScalar(mlx.Zeros(scores.DType(), int(B), int(L), int(folded)), 1)
	scores = mlx.Concatenate([]*mlx.Array{scores, ones}, 2)
	return inds, scores
}

func (m *SparseMoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	inds, scores := m.route(x, cfg, B, L)
	inds, scores = appendFoldedSharedExperts(inds, scores, cfg, m.FoldedSharedExperts)

	expertOut := m.expertForward(x, inds, cfg)
	y := moeWeightedSum(expertOut, scores, x.DType())
	if m.FoldedSharedExperts > 0 {
		return mlx.Reshape(y, B, L, cfg.HiddenSize)
	}

	sharedUp := m.SharedUp.Forward(x)
	sharedHidden := mlx.ReLUSquared(sharedUp)
	shared := m.SharedDown.Forward(sharedHidden)
	y = mlx.Add(y, shared)
	return mlx.Reshape(y, B, L, cfg.HiddenSize)
}

// moeWeightedSum reduces per-expert outputs [B, L, topK, H] by their scores
// [B, L, topK]. A matmul with a one-row left operand, so it rides MLX's
// existing matmul kernels instead of materializing the elementwise product.
func moeWeightedSum(expertOut, scores *mlx.Array, dtype mlx.DType) *mlx.Array {
	scores = mlx.ExpandDims(scores.AsType(expertOut.DType()), 2)
	return mlx.Squeeze(mlx.Matmul(scores, expertOut), 2).AsType(dtype)
}

func (m *SparseMoE) route(x *mlx.Array, cfg *Config, B, L int32) (*mlx.Array, *mlx.Array) {
	routeDType := mlx.DTypeFloat32

	routerWeight := mlx.Transpose(m.RouterWeight.AsType(routeDType), 1, 0)
	logits := mlx.Matmul(x.AsType(routeDType), routerWeight)

	var probs, selection, negSelection *mlx.Array
	if m.CorrectionBias != nil {
		// The fused helper returns both the unbiased sigmoid probabilities
		// used for scores and the negated bias-corrected values used for top-k.
		probs, negSelection = mlx.SigmoidRouter(logits, m.CorrectionBias.AsType(routeDType))
	} else {
		probs = mlx.Sigmoid(logits)
		selection = probs
	}
	if cfg.ExpertGroupCount > 1 && cfg.ExpertGroupUsedCount > 0 && cfg.ExpertGroupUsedCount < cfg.ExpertGroupCount {
		if selection == nil {
			selection = mlx.Neg(negSelection)
		}
		expertsPerGroup := cfg.NRoutedExperts / cfg.ExpertGroupCount
		grouped := mlx.Reshape(selection, B, L, cfg.ExpertGroupCount, expertsPerGroup)
		topPerGroup := expertsPerGroup
		if topPerGroup > 2 {
			topPerGroup = 2
		}
		if topPerGroup > 0 {
			groupInds := mlx.Argpartition(mlx.Neg(grouped), int(topPerGroup)-1, -1)
			groupInds = mlx.SliceStartStop(groupInds, []int32{0, 0, 0, 0}, []int32{B, L, cfg.ExpertGroupCount, topPerGroup})
			groupScores := mlx.Sum(mlx.TakeAlongAxis(grouped, groupInds, -1), -1, false)
			activeGroups := mlx.Argpartition(mlx.Neg(groupScores), int(cfg.ExpertGroupUsedCount)-1, -1)
			activeGroups = mlx.SliceStartStop(activeGroups, []int32{0, 0, 0}, []int32{B, L, cfg.ExpertGroupUsedCount})
			mask := mlx.Zeros(selection.DType(), int(B), int(L), int(cfg.ExpertGroupCount))
			values := mlx.AddScalar(mlx.Zeros(selection.DType(), int(B), int(L), int(cfg.ExpertGroupUsedCount)), 1)
			mask = mask.PutAlongAxis(activeGroups, values, -1)
			mask = mlx.Reshape(
				mlx.Tile(mlx.ExpandDims(mask, -1), []int32{1, 1, 1, expertsPerGroup}),
				B, L, cfg.NRoutedExperts,
			)
			selection = mlx.Mul(selection, mask)
			negSelection = nil
		}
	}

	if negSelection == nil {
		negSelection = mlx.Neg(selection)
	}
	inds := mlx.Argpartition(negSelection, int(cfg.NumExpertsPerTok)-1, -1)
	inds = mlx.SliceStartStop(inds, []int32{0, 0, 0}, []int32{B, L, cfg.NumExpertsPerTok})

	scores := mlx.TakeAlongAxis(probs, inds, -1)
	if cfg.NormTopKProb && cfg.NumExpertsPerTok > 1 {
		sumScores := mlx.AddScalar(mlx.Sum(scores, 2, true), 1e-20)
		scores = mlx.Div(scores, sumScores)
	}
	if cfg.RoutedScalingFactor != 1 {
		scores = mlx.MulScalar(scores, cfg.RoutedScalingFactor)
	}

	return inds, scores
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	residual := x
	h := l.Norm.Forward(x, cfg.LayerNormEpsilon)
	switch l.Type {
	case 'M':
		h = l.Mamba.Forward(h, b, c, B, L, cfg)
	case '*', 'A':
		h = l.Attention.Forward(h, b, c, B, L, cfg)
	case '-':
		h = l.Dense.Forward(h)
	case 'E':
		h = l.MoE.Forward(h, cfg)
	}
	if h.DType() != residual.DType() {
		h = h.AsType(residual.DType())
	}
	out := mlx.Add(residual, h)
	return out
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	tokens := b.InputIDs
	dims := tokens.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	h := m.EmbedTokens.Forward(tokens)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, B, L, m.Config)
	}
	h = m.Norm.Forward(h, m.LayerNormEpsilon)
	return h, h
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(x)
}

// mtpDraft is the model viewed as its own draft: the same struct, carrying
// the DraftModel method set for the inline MTP head.
type mtpDraft Model

// SelfDraft returns the model's drafting view, or nil when no MTP head was
// loaded. The methods exist either way, so this is what decides availability.
func (m *Model) SelfDraft() base.DraftModel {
	if m.MTP == nil {
		return nil
	}
	return (*mtpDraft)(m)
}

// LoadWeights does nothing: an inline head's weights load with the target's.
func (m *mtpDraft) LoadWeights(map[string]*mlx.Array) error { return nil }

func (m *mtpDraft) Unembed(x *mlx.Array) *mlx.Array { return (*Model)(m).Unembed(x) }

// NewCaches builds one slot per MTP layer that keeps state; layers that keep
// none (MoE, dense) contribute nothing.
func (m *mtpDraft) NewCaches() []cache.Cache {
	if m.MTP == nil {
		return nil
	}
	caches := make([]cache.Cache, 0, mtpCacheCount(m.MTP))
	for _, layer := range m.MTP.Layers {
		if c := newMTPLayerCache(layer); c != nil {
			caches = append(caches, c)
		}
	}
	return caches
}

// Forward runs one MTP step; the head's hidden also serves as the aux hidden
// seeding the next step.
func (m *mtpDraft) Forward(b *batch.Batch, _, draftCaches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	emb := m.MTP.Enorm.Forward(m.EmbedTokens.Forward(b.InputIDs), m.LayerNormEpsilon)
	h := m.MTP.Hnorm.Forward(b.Hidden, m.LayerNormEpsilon)
	fused := m.MTP.FC.Forward(emb.Concatenate(-1, h))

	cacheIndex := 0
	for _, layer := range m.MTP.Layers {
		var c cache.Cache
		if mtpLayerUsesCache(layer) && cacheIndex < len(draftCaches) {
			c = draftCaches[cacheIndex]
			cacheIndex++
		}
		fused = layer.Forward(fused, b, c, B, L, m.Config)
	}
	hidden = m.MTP.Norm.Forward(fused, m.LayerNormEpsilon)
	return hidden, hidden
}

func (m *Model) NumLayers() int {
	return len(m.Layers)
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer {
	return m.tok
}

func (m *Model) MaxContextLength() int {
	return int(m.MaxPositionEmbeddings)
}

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i, layer := range m.Layers {
		caches[i] = newLayerCache(layer, m.Config)
	}
	return caches
}

func newLayerCache(layer *Layer, cfg *Config) cache.Cache {
	if layer == nil {
		return nil
	}
	switch layer.Type {
	case 'M':
		return cache.NewRecurrentCache(cfg.ConvKernel-1, cfgConvDim(cfg), cfg.MambaNumHeads, cfg.MambaHeadDim, cfg.SSMStateSize)
	case '*', 'A':
		return cache.NewKVCache()
	default:
		return nil
	}
}

func mtpLayerUsesCache(layer *Layer) bool {
	return layer != nil && (layer.Type == '*' || layer.Type == 'A')
}

func mtpCacheCount(head *MTPHead) int {
	if head == nil {
		return 0
	}
	count := 0
	for _, layer := range head.Layers {
		if mtpLayerUsesCache(layer) {
			count++
		}
	}
	return count
}

func newMTPLayerCache(layer *Layer) cache.Cache {
	if mtpLayerUsesCache(layer) {
		return cache.NewKVCache()
	}
	return nil
}

func cfgConvDim(cfg *Config) int32 {
	inner := cfg.MambaNumHeads * cfg.MambaHeadDim
	return inner + 2*cfg.NGroups*cfg.SSMStateSize
}
