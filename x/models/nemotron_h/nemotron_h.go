// Package nemotron_h provides the Nemotron-H text model implementation for MLX.
package nemotron_h

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
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

var _ base.Model = (*Model)(nil)

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

	tok *tokenizer.Tokenizer
	*Config

	weightPrefix string
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
	ConvWeight *mlx.Array
	ConvBias   *mlx.Array
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
	stacked := mlx.Stack(parts, 0)
	cloned := stacked.Clone()
	mlx.Eval(cloned)
	return cloned
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

func transposeExpertWeightForGatherMM(w *mlx.Array) *mlx.Array {
	if w == nil || !w.Valid() || w.NumDims() != 3 {
		return w
	}
	t := mlx.Transpose(w, 0, 2, 1)
	cloned := t.Clone()
	mlx.Eval(cloned)
	return cloned
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
	out := mlx.Concatenate([]*mlx.Array{dst, src}, 0).Clone()
	mlx.Eval(out)
	return out
}

func stackSlicesAndClone(slices []*mlx.Array) *mlx.Array {
	if len(slices) == 0 {
		return nil
	}
	out := mlx.Stack(slices, 0).Clone()
	mlx.Eval(out)
	return out
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
		if useQuantized && supportsGatherQMM(m, b) {
			weights = append(weights, w)
			scales = append(scales, scale)
			if qbias != nil {
				biases = append(biases, qbias)
			}
			continue
		}

		weights = append(weights, mlx.Dequantize(w, scale, qbias, gs, b, m))
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
		layer := m.Layers[i]
		if layer == nil {
			layer = &Layer{Type: cfg.LayerTypes[i]}
		}

		norm := tensors[layerPrefix+".norm.weight"]
		if norm == nil {
			return fmt.Errorf("layer %d: missing norm weight", i)
		}
		layer.Norm = nn.NewRMSNorm(norm, cfg.LayerNormEpsilon)

		switch layer.Type {
		case 'M':
			mixerPrefix := layerPrefix + ".mixer"
			mamba := &Mamba2{}
			mamba.InProj = linears.Make(mixerPrefix + ".in_proj")
			mamba.OutProj = linears.Make(mixerPrefix + ".out_proj")
			mamba.ConvWeight = sanitizeConvWeight(tensors[mixerPrefix+".conv1d.weight"])
			mamba.ConvBias = tensors[mixerPrefix+".conv1d.bias"]
			mamba.DtBias = tensors[mixerPrefix+".dt_bias"]
			aLog := tensors[mixerPrefix+".A_log"]
			if aLog != nil {
				mamba.A = mlx.Neg(mlx.Exp(aLog.AsType(mlx.DTypeFloat32)))
			}
			mamba.D = tensors[mixerPrefix+".D"]
			mamba.NormWeight = tensors[mixerPrefix+".norm.weight"]
			if mamba.InProj == nil || mamba.OutProj == nil || mamba.ConvWeight == nil || mamba.DtBias == nil || mamba.A == nil || mamba.D == nil || mamba.NormWeight == nil {
				return fmt.Errorf("layer %d: missing mamba2 tensors", i)
			}
			if mamba.ConvWeight.NumDims() != 2 {
				return fmt.Errorf("layer %d: conv1d weight must be 2D after sanitization, got %dD", i, mamba.ConvWeight.NumDims())
			}
			mamba.Conv1D = nn.NewConv1d(mlx.ExpandDims(mamba.ConvWeight, 2), nil, 1, 0, 1, int32(mamba.ConvWeight.Dim(0)))
			layer.Mamba = mamba
		case '*', 'A':
			mixerPrefix := layerPrefix + ".mixer"
			attn := &Attention{
				QProj: linears.Make(mixerPrefix + ".q_proj"),
				KProj: linears.Make(mixerPrefix + ".k_proj"),
				VProj: linears.Make(mixerPrefix + ".v_proj"),
				OProj: linears.Make(mixerPrefix + ".o_proj"),
			}
			if attn.QProj == nil || attn.KProj == nil || attn.VProj == nil || attn.OProj == nil {
				return fmt.Errorf("layer %d: missing attention projections", i)
			}
			layer.Attention = attn
		case '-':
			mixerPrefix := layerPrefix + ".mixer"
			dense := &DenseMLP{
				UpProj:   linears.Make(mixerPrefix + ".up_proj"),
				DownProj: linears.Make(mixerPrefix + ".down_proj"),
			}
			if dense.UpProj == nil || dense.DownProj == nil {
				return fmt.Errorf("layer %d: missing dense mlp projections", i)
			}
			layer.Dense = dense
		case 'E':
			mixerPrefix := layerPrefix + ".mixer"
			moe := &SparseMoE{
				Router:         linears.Make(mixerPrefix + ".gate"),
				RouterWeight:   tensors[mixerPrefix+".gate.weight"],
				CorrectionBias: tensors[mixerPrefix+".gate.e_score_correction_bias"],
				SharedUp:       linears.Make(mixerPrefix + ".shared_experts.up_proj"),
				SharedDown:     linears.Make(mixerPrefix + ".shared_experts.down_proj"),
			}
			if moe.Router == nil || moe.RouterWeight == nil || moe.SharedUp == nil || moe.SharedDown == nil {
				return fmt.Errorf("layer %d: missing moe router or shared expert projections", i)
			}
			if cfg.NRoutedExperts <= 0 || cfg.NumExpertsPerTok <= 0 {
				return fmt.Errorf("layer %d: invalid moe config", i)
			}
			up := collectExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "up_proj", cfg.NRoutedExperts)
			down := collectExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "down_proj", cfg.NRoutedExperts)
			if up == nil || down == nil {
				return fmt.Errorf("layer %d: missing routed expert weights", i)
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
		}

		m.Layers[i] = layer
	}
	return nil
}

func reluSquared(x *mlx.Array) *mlx.Array {
	zero := mlx.NewScalarArray(float32(0)).AsType(x.DType())
	x = mlx.Maximum(x, zero)
	return mlx.Mul(x, x)
}

func depthwiseCausalConv1d(x, w *mlx.Array, outLen int32) *mlx.Array {
	if x == nil || w == nil || w.NumDims() != 2 {
		return nil
	}
	B := int32(x.Dim(0))
	C := int32(w.Dim(0))
	K := int32(w.Dim(1))
	var out *mlx.Array
	for i := range K {
		seg := mlx.SliceStartStop(x, []int32{0, i, 0}, []int32{B, i + outLen, C})
		wi := mlx.SliceStartStop(w, []int32{0, i}, []int32{C, i + 1})
		wi = mlx.Reshape(wi, 1, 1, C)
		term := mlx.Mul(seg, wi)
		if out == nil {
			out = term
		} else {
			out = mlx.Add(out, term)
		}
	}
	return out
}

func repeatGroups(x *mlx.Array, repeats int32) *mlx.Array {
	if repeats <= 1 {
		return x
	}
	// Mamba2 maps each B/C group to a contiguous block of heads.
	dims := x.Dims()
	x = mlx.ExpandDims(x, 3)
	x = mlx.Tile(x, []int32{1, 1, 1, repeats, 1})
	return mlx.Reshape(x, int32(dims[0]), int32(dims[1]), int32(dims[2])*repeats, int32(dims[3]))
}

func sliceTime(x *mlx.Array, t int32) *mlx.Array {
	dims := x.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	for i, d := range dims {
		stop[i] = int32(d)
	}
	start[1] = t
	stop[1] = t + 1
	return mlx.Squeeze(mlx.SliceStartStop(x, start, stop), 1)
}

func mamba2LoopScan(hidden, bState, cState, dt, state, a, d, dtBias *mlx.Array, B, L int32, cfg *Config, splits []int, aggressiveDetach bool) (*mlx.Array, *mlx.Array, []*mlx.Array) {
	outs := make([]*mlx.Array, 0, L)
	deltaStates := make([]*mlx.Array, 0, len(splits)+1)
	splitIdx := 0
	for t := range L {
		xt := sliceTime(hidden, t).AsType(mlx.DTypeFloat32)
		bt := sliceTime(bState, t).AsType(mlx.DTypeFloat32)
		ct := sliceTime(cState, t).AsType(mlx.DTypeFloat32)
		dtt := sliceTime(dt, t).AsType(mlx.DTypeFloat32)
		dtt = mlx.Add(dtt, dtBias)
		dtt = mlx.Exp(dtt)
		dtt = mlx.AddScalar(dtt, 1)
		dtt = mlx.Log(dtt)

		reshapedDtt := mlx.Reshape(dtt, B, cfg.MambaNumHeads, 1, 1)
		product := mlx.Mul(reshapedDtt, a)
		dA := mlx.Exp(product)
		dB := mlx.Mul(mlx.Reshape(dtt, B, cfg.MambaNumHeads, 1), bt)
		dBx := mlx.Mul(mlx.ExpandDims(xt, -1), mlx.ExpandDims(dB, 2))
		state = mlx.Add(mlx.Mul(state, dA), dBx)
		if aggressiveDetach {
			state = state.Clone()
			mlx.Eval(state)
		}

		y := mlx.Sum(mlx.Mul(state, mlx.ExpandDims(ct, 2)), 3, false)
		y = mlx.Add(y, mlx.Mul(xt, d))
		if aggressiveDetach {
			y = y.Clone()
			mlx.Eval(y)
		}
		outs = append(outs, y)
		if splitIdx < len(splits) && int(t+1) == splits[splitIdx] {
			deltaStates = append(deltaStates, state)
			splitIdx++
		}
	}

	return mlx.Stack(outs, 1), state, deltaStates
}

// mamba2ScanWithSnapshots uses optional fused scan helpers when they support
// the current backend and shapes. It returns ok=false when the caller should
// use the backend-neutral loop fallback.
func mamba2ScanWithSnapshots(hidden, bState, cState, dt, state, a, d, dtBias *mlx.Array, B, L int32, cfg *Config, splits []int) (*mlx.Array, *mlx.Array, []*mlx.Array, bool) {
	if len(splits) == 1 && splits[0] > 0 && splits[0] < int(L) {
		y, nextState, snapshotState, ok := mlx.FastMamba2ScanWithSnapshot(hidden, bState, cState, dt, state, a, d, dtBias, splits[0])
		if ok {
			return y, nextState, []*mlx.Array{snapshotState}, true
		}
	}

	outs := make([]*mlx.Array, 0, len(splits)+1)
	deltaStates := make([]*mlx.Array, 0, len(splits)+1)
	scanState := state
	start := int32(0)
	groups := int32(bState.Dim(2))

	runSegment := func(end int32, capture bool) bool {
		if end <= start || end > L {
			return false
		}
		segHidden := mlx.SliceStartStop(hidden, []int32{0, start, 0, 0}, []int32{B, end, cfg.MambaNumHeads, cfg.MambaHeadDim}).AsType(mlx.DTypeFloat32)
		segB := mlx.SliceStartStop(bState, []int32{0, start, 0, 0}, []int32{B, end, groups, cfg.SSMStateSize}).AsType(mlx.DTypeFloat32)
		segC := mlx.SliceStartStop(cState, []int32{0, start, 0, 0}, []int32{B, end, groups, cfg.SSMStateSize}).AsType(mlx.DTypeFloat32)
		segDt := mlx.SliceStartStop(dt, []int32{0, start, 0}, []int32{B, end, cfg.MambaNumHeads}).AsType(mlx.DTypeFloat32)

		y, nextState, ok := mlx.FastMamba2Scan(segHidden, segB, segC, segDt, scanState, a, d, dtBias)
		if !ok {
			return false
		}
		outs = append(outs, y)
		scanState = nextState
		if capture {
			deltaStates = append(deltaStates, scanState)
		}
		start = end
		return true
	}

	for _, split := range splits {
		if !runSegment(int32(split), true) {
			return nil, nil, nil, false
		}
	}
	if !runSegment(L, false) {
		return nil, nil, nil, false
	}

	if len(outs) == 1 {
		return outs[0], scanState, deltaStates, true
	}
	return mlx.Concatenate(outs, 1), scanState, deltaStates, true
}

func (m *Mamba2) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	dtype := x.DType()
	aggressiveDetach := c == nil
	projected := m.InProj.Forward(x)
	inner := cfg.MambaNumHeads * cfg.MambaHeadDim
	convDim := inner + 2*cfg.NGroups*cfg.SSMStateSize

	gate := mlx.SliceStartStop(projected, []int32{0, 0, 0}, []int32{B, L, inner})
	if aggressiveDetach {
		gate = gate.Clone()
		mlx.Eval(gate)
	}
	xBC := mlx.SliceStartStop(projected, []int32{0, 0, inner}, []int32{B, L, inner + convDim}).AsType(mlx.DTypeFloat32)
	dt := mlx.SliceStartStop(projected, []int32{0, 0, inner + convDim}, []int32{B, L, inner + convDim + cfg.MambaNumHeads})
	if aggressiveDetach {
		dt = dt.Clone()
		mlx.Eval(dt)
	}

	convTail := cfg.ConvKernel - 1
	var rc *cache.RecurrentCache
	var convState *mlx.Array
	var state *mlx.Array
	var splits []int
	if c != nil {
		if typed, ok := c.(*cache.RecurrentCache); ok {
			rc = typed
			history := rc.Get(b, mlx.DTypeFloat32)
			convState = history.ConvState()
			state = history.DeltaState()
			splits = rc.SnapshotSplits(int(L))
		}
	}
	if convState == nil {
		convState = mlx.Zeros(mlx.DTypeFloat32, int(B), int(convTail), int(convDim))
	}
	if state == nil {
		state = mlx.Zeros(mlx.DTypeFloat32, int(B), int(cfg.MambaNumHeads), int(cfg.MambaHeadDim), int(cfg.SSMStateSize))
	}

	convInput := mlx.Concatenate([]*mlx.Array{convState, xBC}, 1)
	convWeight := m.ConvWeight.AsType(mlx.DTypeFloat32)
	var convOut *mlx.Array
	if m.ConvBias != nil {
		if customConvOut, ok := mlx.FastMambaDepthwiseConvSiLU(convInput, convWeight, m.ConvBias.AsType(mlx.DTypeFloat32), int(L)); ok {
			convOut = customConvOut
		}
	}
	if convOut == nil {
		if m.Conv1D != nil {
			convOut = m.Conv1D.Forward(convInput)
		} else {
			convOut = depthwiseCausalConv1d(convInput, convWeight, L)
		}
		if m.ConvBias != nil {
			convOut = mlx.Add(convOut, m.ConvBias.AsType(mlx.DTypeFloat32))
		}
		convOut = mlx.SiLU(convOut)
	}
	if dtype != mlx.DTypeFloat32 {
		convOut = convOut.AsType(dtype).AsType(mlx.DTypeFloat32)
	}

	hidden := mlx.SliceStartStop(convOut, []int32{0, 0, 0}, []int32{B, L, inner})
	bState := mlx.SliceStartStop(convOut, []int32{0, 0, inner}, []int32{B, L, inner + cfg.NGroups*cfg.SSMStateSize})
	cState := mlx.SliceStartStop(convOut, []int32{0, 0, inner + cfg.NGroups*cfg.SSMStateSize}, []int32{B, L, inner + 2*cfg.NGroups*cfg.SSMStateSize})

	hidden = mlx.Reshape(hidden, B, L, cfg.MambaNumHeads, cfg.MambaHeadDim)
	bState = mlx.Reshape(bState, B, L, cfg.NGroups, cfg.SSMStateSize)
	cState = mlx.Reshape(cState, B, L, cfg.NGroups, cfg.SSMStateSize)
	if aggressiveDetach {
		hidden = hidden.Clone()
		bState = bState.Clone()
		cState = cState.Clone()
		mlx.Eval(hidden, bState, cState)
	}

	var y *mlx.Array
	var deltaStates []*mlx.Array
	if customY, customState, customDeltaStates, ok := mamba2ScanWithSnapshots(
		hidden,
		bState,
		cState,
		dt.AsType(mlx.DTypeFloat32),
		state,
		mlx.Reshape(m.A.AsType(mlx.DTypeFloat32), cfg.MambaNumHeads),
		mlx.Reshape(m.D.AsType(mlx.DTypeFloat32), cfg.MambaNumHeads),
		mlx.Reshape(m.DtBias.AsType(mlx.DTypeFloat32), cfg.MambaNumHeads),
		B,
		L,
		cfg,
		splits,
	); ok {
		y, state = customY, customState
		deltaStates = customDeltaStates
	}
	if y == nil {
		repeats := cfg.MambaNumHeads / cfg.NGroups
		bState = repeatGroups(bState, repeats)
		cState = repeatGroups(cState, repeats)
		if aggressiveDetach {
			bState = bState.Clone()
			cState = cState.Clone()
			mlx.Eval(bState, cState)
		}
		a := mlx.Reshape(m.A, 1, cfg.MambaNumHeads, 1, 1)
		d := mlx.Reshape(m.D.AsType(mlx.DTypeFloat32), 1, cfg.MambaNumHeads, 1)
		dtBias := mlx.Tile(mlx.Reshape(m.DtBias.AsType(mlx.DTypeFloat32), 1, cfg.MambaNumHeads), []int32{B, 1})
		y, state, deltaStates = mamba2LoopScan(hidden, bState, cState, dt, state, a, d, dtBias, B, L, cfg, splits, aggressiveDetach)
	}
	y = mlx.Reshape(y, B, L, inner)
	y = gatedGroupRMSNormAsType(y, gate, m.NormWeight, cfg, dtype)
	out := m.OutProj.Forward(y)
	if rc != nil {
		convStates := make([]*mlx.Array, 0, len(splits)+1)
		for _, split := range splits {
			st := mambaConvStateAt(convInput, b.SeqQueryLens, convTail, int32(split))
			if L > 1 {
				st = mlx.Contiguous(st, false)
			}
			convStates = append(convStates, st)
		}
		st := mambaConvStateAt(convInput, b.SeqQueryLens, convTail, L)
		if L > 1 {
			st = mlx.Contiguous(st, false)
		}
		convStates = append(convStates, st)
		deltaStates = append(deltaStates, state)
		rc.Put(b, convStates, deltaStates)
	}
	return out
}

func mambaConvStateAt(concat *mlx.Array, queryLens []int32, convTail, boundary int32) *mlx.Array {
	B := int32(concat.Dim(0))
	D := int32(concat.Dim(2))

	if convTail > 0 && slices.ContainsFunc(queryLens, func(q int32) bool { return boundary > q }) {
		offsets := make([]int32, int(B*convTail))
		for i := range int(B) {
			end := min(boundary, queryLens[i])
			for k := range int(convTail) {
				offsets[i*int(convTail)+k] = end + int32(k)
			}
		}
		positions := mlx.NewArrayInt32(offsets, []int32{B, convTail, 1})
		return mlx.TakeAlongAxis(concat, positions, 1)
	}

	return mlx.SliceStartStop(concat,
		[]int32{0, boundary, 0},
		[]int32{B, boundary + convTail, D})
}

func gatedGroupRMSNormAsType(y, gate, weight *mlx.Array, cfg *Config, dtype mlx.DType) *mlx.Array {
	if out, ok := mlx.FastMambaGatedGroupRMSNorm(y, gate, weight, int(cfg.NGroups), cfg.LayerNormEpsilon, dtype); ok {
		return out
	}
	return gatedGroupRMSNormFallback(y, gate, weight, cfg).AsType(dtype)
}

func gatedGroupRMSNormFallback(y, gate, weight *mlx.Array, cfg *Config) *mlx.Array {
	inner := cfg.MambaNumHeads * cfg.MambaHeadDim
	dims := y.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	y = mlx.Mul(y, mlx.SiLU(gate.AsType(y.DType())))
	groupSize := inner / cfg.NGroups
	y = mlx.Reshape(y, B, L, cfg.NGroups, groupSize)
	variance := mlx.Mean(mlx.Mul(y, y), 3, true)
	y = mlx.Mul(y, mlx.RSqrt(mlx.AddScalar(variance, cfg.LayerNormEpsilon)))
	w := mlx.Reshape(weight.AsType(y.DType()), 1, 1, cfg.NGroups, groupSize)
	y = mlx.Mul(y, w)
	return mlx.Reshape(y, B, L, inner)
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
	hidden := reluSquared(up)
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

func (m *SparseMoE) canMapQuantizedExpert() bool {
	if m.UpWeightQ == nil ||
		m.UpScales == nil ||
		m.UpBiases != nil ||
		m.DownWeightQ == nil ||
		m.DownScales == nil ||
		m.DownBiases != nil {
		return false
	}

	return m.canMapUpProjection() && m.canMapDownProjection()
}

func (m *SparseMoE) canMapUpProjection() bool {
	return mlx.SupportsMoEGatherQMMBlockMapped(m.UpGroupSize, m.UpBits, m.UpMode)
}

func (m *SparseMoE) canMapDownProjection() bool {
	return mlx.SupportsMoEGatherQMMBlockMapped(m.DownGroupSize, m.DownBits, m.DownMode)
}

func (m *SparseMoE) mappedUpProjection(xFlat *mlx.Array, expertMap *mlx.MoEGatherQMMMap, topK int) (*mlx.Array, bool) {
	return mlx.FastMoEGatherQMMBlockMapped(xFlat, m.UpWeightQ, m.UpScales, expertMap, topK, m.UpGroupSize, m.UpBits, m.UpMode)
}

func (m *SparseMoE) mappedDownProjection(hidden *mlx.Array, expertMap *mlx.MoEGatherQMMMap, topK int, applyReluSquared bool) (*mlx.Array, bool) {
	if applyReluSquared {
		return mlx.FastMoEGatherQMMBlockMappedReLUSquared(hidden, m.DownWeightQ, m.DownScales, expertMap, topK, m.DownGroupSize, m.DownBits, m.DownMode)
	}
	return mlx.FastMoEGatherQMMBlockMapped(hidden, m.DownWeightQ, m.DownScales, expertMap, topK, m.DownGroupSize, m.DownBits, m.DownMode)
}

func (m *SparseMoE) mappedQuantizedExpertForward(x, indices *mlx.Array, cfg *Config) (*mlx.Array, bool) {
	if !m.canMapQuantizedExpert() {
		return nil, false
	}

	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := int32(indices.Dim(2))
	tokens := B * L
	if !shouldSortMoEExperts(tokens) {
		return nil, false
	}
	idxFlat := mlx.Reshape(indices, tokens, topK)

	expertMap, ok := mlx.NewMoEGatherQMMMap(idxFlat, m.UpWeightQ.Dim(0))
	if !ok {
		return nil, false
	}

	xFlat := mlx.Reshape(x, tokens, 1, cfg.HiddenSize)
	up, ok := m.mappedUpProjection(xFlat, expertMap, int(topK))
	if !ok {
		return nil, false
	}
	down, ok := m.mappedDownProjection(up, expertMap, int(topK), true)
	if !ok {
		hidden := reluSquared(up)
		if hidden.DType() != up.DType() {
			hidden = hidden.AsType(up.DType())
		}
		down, ok = m.mappedDownProjection(hidden, expertMap, int(topK), false)
	}
	if !ok {
		return nil, false
	}

	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize), true
}

func (m *SparseMoE) expertForward(x *mlx.Array, indices *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := int32(indices.Dim(2))
	if out, ok := m.mappedQuantizedExpertForward(x, indices, cfg); ok {
		return out
	}

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

	hidden := reluSquared(up)
	if hidden.DType() != up.DType() {
		// Keep the activation in the projection dtype so dense and quantized
		// down projections stay on the BF16 fast path.
		hidden = hidden.AsType(up.DType())
	}
	down := m.gatherExpertDown(hidden, idxFlat, doSort)

	var out *mlx.Array
	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}
	out = mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
	return out
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
	sharedHidden := reluSquared(sharedUp)
	shared := m.SharedDown.Forward(sharedHidden)
	y = mlx.Add(y, shared)
	return mlx.Reshape(y, B, L, cfg.HiddenSize)
}

func moeWeightedSum(expertOut, scores *mlx.Array, dtype mlx.DType) *mlx.Array {
	if y, ok := mlx.FastMoEWeightedSum(expertOut, scores, dtype); ok {
		return y
	}
	return mlx.Sum(mlx.Mul(expertOut, mlx.ExpandDims(scores.AsType(expertOut.DType()), -1)), 2, false).AsType(dtype)
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

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	tokens := b.InputIDs
	dims := tokens.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	h := m.EmbedTokens.Forward(tokens)
	noCacheDetach := caches == nil
	if noCacheDetach {
		h = h.Clone()
		mlx.Eval(h)
	}
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, B, L, m.Config)
		if noCacheDetach {
			h = h.Clone()
			mlx.Eval(h)
		}
	}
	h = m.Norm.Forward(h, m.LayerNormEpsilon)
	return h
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(x)
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
	convTail := m.ConvKernel - 1
	convDim := cfgConvDim(m.Config)
	for i, layer := range m.Layers {
		switch layer.Type {
		case 'M':
			caches[i] = cache.NewRecurrentCache(convTail, convDim, m.MambaNumHeads, m.MambaHeadDim, m.SSMStateSize)
		case '*', 'A':
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

func cfgConvDim(cfg *Config) int32 {
	inner := cfg.MambaNumHeads * cfg.MambaHeadDim
	return inner + 2*cfg.NGroups*cfg.SSMStateSize
}
