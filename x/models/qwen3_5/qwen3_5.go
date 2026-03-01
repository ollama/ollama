//go:build mlx

// Package qwen3_5 provides the Qwen 3.5 text and MoE implementation for MLX.
package qwen3_5

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("Qwen3_5ForCausalLM", NewModel)
	base.Register("Qwen3_5ForConditionalGeneration", NewModel)
	base.Register("Qwen3NextForCausalLM", NewModel)
	base.Register("Qwen3NextForConditionalGeneration", NewModel)
}

// RopeParameters carries optional rope metadata embedded under rope_parameters.
type RopeParameters struct {
	Type                string  `json:"type"`
	RopeType            string  `json:"rope_type"`
	RopeTheta           float32 `json:"rope_theta"`
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
}

// Config holds Qwen 3.5 text config (top-level or nested text_config).
type Config struct {
	ModelType             string   `json:"model_type"`
	HiddenSize            int32    `json:"hidden_size"`
	IntermediateSize      int32    `json:"intermediate_size"`
	NumHiddenLayers       int32    `json:"num_hidden_layers"`
	NumAttentionHeads     int32    `json:"num_attention_heads"`
	NumKeyValueHeads      int32    `json:"num_key_value_heads"`
	HeadDim               int32    `json:"head_dim"`
	RMSNormEps            float32  `json:"rms_norm_eps"`
	VocabSize             int32    `json:"vocab_size"`
	MaxPositionEmbeddings int32    `json:"max_position_embeddings"`
	AttentionBias         bool     `json:"attention_bias"`
	TieWordEmbeddings     bool     `json:"tie_word_embeddings"`
	LayerTypes            []string `json:"layer_types"`

	FullAttentionInterval int32 `json:"full_attention_interval"`

	LinearNumValueHeads  int32 `json:"linear_num_value_heads"`
	LinearNumKeyHeads    int32 `json:"linear_num_key_heads"`
	LinearKeyHeadDim     int32 `json:"linear_key_head_dim"`
	LinearValueHeadDim   int32 `json:"linear_value_head_dim"`
	LinearConvKernelDim  int32 `json:"linear_conv_kernel_dim"`
	DecoderSparseStep    int32 `json:"decoder_sparse_step"`
	SharedExpertGateRank int32 `json:"-"`

	NumExperts                   int32   `json:"num_experts"`
	NumExpertsPerTok             int32   `json:"num_experts_per_tok"`
	SharedExpertIntermediateSize int32   `json:"shared_expert_intermediate_size"`
	MoeIntermediateSize          int32   `json:"moe_intermediate_size"`
	NormTopKProb                 bool    `json:"norm_topk_prob"`
	MLPOnlyLayers                []int32 `json:"mlp_only_layers"`

	RopeTheta           float32         `json:"rope_theta"`
	PartialRotaryFactor float32         `json:"partial_rotary_factor"`
	RopeScaling         map[string]any  `json:"rope_scaling"`
	RopeParameters      *RopeParameters `json:"rope_parameters"`

	// Quantization metadata.
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	Scale   float32 `json:"-"`
	RopeDim int32   `json:"-"`
}

// Model is the Qwen 3.5 model.
type Model struct {
	EmbedTokens *nn.Embedding
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config

	weightPrefix string
}

// Layer is a transformer decoder layer.
type Layer struct {
	InputNorm         *nn.RMSNorm
	PostAttentionNorm *nn.RMSNorm

	IsLinear bool
	FullAttn *FullAttention
	Linear   *GatedDeltaNet
	MLP      MLPBlock
}

// FullAttention is the full-attention branch used every N layers.
type FullAttention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer

	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm
}

// GatedDeltaNet is the recurrent linear-attention branch.
type GatedDeltaNet struct {
	InProjQKV  nn.LinearLayer
	InProjZ    nn.LinearLayer
	InProjB    nn.LinearLayer
	InProjA    nn.LinearLayer
	InProjQKVZ nn.LinearLayer
	InProjBA   nn.LinearLayer
	OutProj    nn.LinearLayer

	Conv1D     *nn.Conv1d
	ConvWeight *mlx.Array
	NormWeight *mlx.Array
	DtBias     *mlx.Array
	ALog       *mlx.Array
	AExp       *mlx.Array
}

// MLPBlock is the feed-forward interface for dense and MoE blocks.
type MLPBlock interface {
	Forward(x *mlx.Array, cfg *Config) *mlx.Array
}

// DenseMLP is SwiGLU feed-forward.
type DenseMLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

// SparseMoE is Qwen3.5's sparse MoE with shared expert.
type SparseMoE struct {
	Gate             nn.LinearLayer
	SwitchMLP        *SwitchMLP
	SharedExpert     *DenseMLP
	SharedExpertGate nn.LinearLayer
}

// SwitchMLP executes selected expert MLPs.
type SwitchMLP struct {
	GateWeight *mlx.Array
	UpWeight   *mlx.Array
	DownWeight *mlx.Array

	GateWeightQ, GateScales, GateBiases *mlx.Array
	UpWeightQ, UpScales, UpBiases       *mlx.Array
	DownWeightQ, DownScales, DownBiases *mlx.Array

	GateBits int
	UpBits   int
	DownBits int

	GateGroupSize int
	UpGroupSize   int
	DownGroupSize int

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
	var rawTop map[string]json.RawMessage
	if err := json.Unmarshal(configData, &rawTop); err != nil {
		return Config{}, fmt.Errorf("parse config envelope: %w", err)
	}

	var cfg Config
	activeRaw := rawTop
	if textRaw, ok := rawTop["text_config"]; ok {
		if err := json.Unmarshal(textRaw, &cfg); err != nil {
			return Config{}, fmt.Errorf("parse text_config: %w", err)
		}
		if err := json.Unmarshal(textRaw, &activeRaw); err != nil {
			return Config{}, fmt.Errorf("parse text_config envelope: %w", err)
		}
	} else {
		if err := json.Unmarshal(configData, &cfg); err != nil {
			return Config{}, fmt.Errorf("parse config: %w", err)
		}
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
	if cfg.HeadDim <= 0 {
		return Config{}, fmt.Errorf("invalid head_dim: %d", cfg.HeadDim)
	}

	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.LinearConvKernelDim <= 0 {
		cfg.LinearConvKernelDim = 4
	}
	if cfg.LinearNumKeyHeads <= 0 || cfg.LinearNumValueHeads <= 0 || cfg.LinearKeyHeadDim <= 0 || cfg.LinearValueHeadDim <= 0 {
		return Config{}, fmt.Errorf("invalid linear attention config (k_heads=%d v_heads=%d k_dim=%d v_dim=%d)",
			cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads, cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim)
	}
	if cfg.LinearNumValueHeads%cfg.LinearNumKeyHeads != 0 {
		return Config{}, fmt.Errorf("linear_num_value_heads (%d) must be divisible by linear_num_key_heads (%d)", cfg.LinearNumValueHeads, cfg.LinearNumKeyHeads)
	}

	if cfg.RopeParameters != nil {
		if cfg.RopeParameters.RopeTheta > 0 {
			cfg.RopeTheta = cfg.RopeParameters.RopeTheta
		}
		if cfg.RopeParameters.PartialRotaryFactor > 0 {
			cfg.PartialRotaryFactor = cfg.RopeParameters.PartialRotaryFactor
		}
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 100000.0
	}
	if cfg.PartialRotaryFactor == 0 {
		cfg.PartialRotaryFactor = 0.25
	}
	if cfg.PartialRotaryFactor < 0 {
		cfg.PartialRotaryFactor = 0.25
	}
	ropeDim := int32(float32(cfg.HeadDim) * cfg.PartialRotaryFactor)
	if ropeDim <= 0 {
		ropeDim = cfg.HeadDim
	}
	if ropeDim > cfg.HeadDim {
		ropeDim = cfg.HeadDim
	}
	cfg.RopeDim = ropeDim

	if cfg.FullAttentionInterval <= 0 {
		for i, lt := range cfg.LayerTypes {
			if strings.Contains(strings.ToLower(lt), "full") {
				cfg.FullAttentionInterval = int32(i + 1)
				break
			}
		}
		if cfg.FullAttentionInterval <= 0 {
			cfg.FullAttentionInterval = 4
		}
	}
	if cfg.FullAttentionInterval > cfg.NumHiddenLayers {
		cfg.FullAttentionInterval = cfg.NumHiddenLayers
	}

	if cfg.NumExperts > 0 {
		if cfg.NumExpertsPerTok <= 0 {
			cfg.NumExpertsPerTok = 1
		}
		if cfg.MoeIntermediateSize <= 0 {
			cfg.MoeIntermediateSize = cfg.IntermediateSize
		}
		if cfg.SharedExpertIntermediateSize <= 0 {
			cfg.SharedExpertIntermediateSize = cfg.IntermediateSize
		}
		if _, ok := activeRaw["norm_topk_prob"]; !ok {
			cfg.NormTopKProb = true
		}
		if cfg.DecoderSparseStep <= 0 {
			cfg.DecoderSparseStep = 1
		}
	}

	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	return cfg, nil
}

type tensorPathLayout struct {
	containerPrefix string
	modelPrefix     string
}

func (l tensorPathLayout) modelPath(suffix string) string {
	return l.containerPrefix + l.modelPrefix + suffix
}

func resolveTensorPathLayout(tensors map[string]*mlx.Array) tensorPathLayout {
	for _, layout := range []tensorPathLayout{
		{containerPrefix: "", modelPrefix: "model."},
		{containerPrefix: "language_model.", modelPrefix: "model."},
		{containerPrefix: "language_model.", modelPrefix: ""},
		{containerPrefix: "model.language_model.", modelPrefix: "model."},
		{containerPrefix: "model.language_model.", modelPrefix: ""},
	} {
		if tensors[layout.modelPath("embed_tokens.weight")] != nil {
			return layout
		}
	}

	return tensorPathLayout{modelPrefix: "model."}
}

func layerIsLinear(cfg *Config, layer int32) bool {
	if len(cfg.LayerTypes) == int(cfg.NumHiddenLayers) {
		t := strings.ToLower(cfg.LayerTypes[layer])
		return !strings.Contains(t, "full")
	}
	if cfg.FullAttentionInterval <= 0 {
		return true
	}
	return (layer+1)%cfg.FullAttentionInterval != 0
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
	if cfg.DecoderSparseStep <= 1 {
		return true
	}
	return (layer+1)%cfg.DecoderSparseStep == 0
}

// NewModel creates a Qwen 3.5 model from a manifest root.
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

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		m.Layers[i] = &Layer{IsLinear: layerIsLinear(&cfg, i)}
	}

	return m, nil
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

func supportsGatherQMM(mode string, bits int) bool {
	return mode == "affine" && (bits == 4 || bits == 8)
}

func freeTensorKeys(tensors map[string]*mlx.Array, keys ...string) {
	for _, k := range keys {
		if k == "" {
			continue
		}
		if t := tensors[k]; t != nil {
			mlx.Unpin(t)
			delete(tensors, k)
		}
	}
}

func stackAndDetach(parts []*mlx.Array) *mlx.Array {
	if len(parts) == 0 {
		return nil
	}
	stacked := mlx.Stack(parts, 0)
	detached := mlx.Detach(stacked)
	mlx.Eval(detached)
	mlx.Unpin(stacked)
	return detached
}

func transposeExpertWeightForGatherMM(w *mlx.Array) *mlx.Array {
	if w == nil || !w.Valid() || w.NumDims() != 3 {
		return w
	}
	t := mlx.Transpose(w, 0, 2, 1)
	d := mlx.Detach(t)
	mlx.Eval(d)
	mlx.Unpin(t)
	return d
}

func describeMoEProjection(prefix string, w *stackedExpertWeights) string {
	if w == nil {
		return prefix + "=missing"
	}
	if w.Scales != nil {
		return fmt.Sprintf("%s=qmm(mode=%s,bits=%d,gs=%d)", prefix, w.Mode, w.Bits, w.GroupSize)
	}
	if w.Bits > 0 || w.Mode != "" {
		reason := "dequantized"
		if !supportsGatherQMM(w.Mode, w.Bits) {
			reason = "unsupported_gather_qmm"
		}
		return fmt.Sprintf("%s=%s(mode=%s,bits=%d,gs=%d)", prefix, reason, w.Mode, w.Bits, w.GroupSize)
	}
	return prefix + "=fp"
}

func summarizeMoEFallbackReason(gateW, upW, downW *stackedExpertWeights) string {
	for _, w := range []*stackedExpertWeights{gateW, upW, downW} {
		if w == nil {
			return "missing_projection"
		}
		if w.Scales != nil {
			continue
		}
		if w.Bits > 0 || w.Mode != "" {
			if !supportsGatherQMM(w.Mode, w.Bits) {
				return fmt.Sprintf("unsupported_gather_qmm(mode=%s,bits=%d)", w.Mode, w.Bits)
			}
			return "dequantized_quant_weights"
		}
	}
	return "unquantized_weights"
}

func sliceStackedExpertAxis1(a *mlx.Array, start, stop int32) *mlx.Array {
	if a == nil || !a.Valid() {
		return nil
	}
	dims := a.Dims()
	if len(dims) < 2 {
		return nil
	}
	beg := make([]int32, len(dims))
	end := make([]int32, len(dims))
	for i, d := range dims {
		end[i] = int32(d)
	}
	beg[1] = start
	end[1] = stop
	return mlx.SliceStartStop(a, beg, end)
}

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
		groupSize, bits, mode := model.ResolveLinearQuantParams(
			cfg.QuantGroupSize,
			cfg.QuantBits,
			cfg.QuantMode,
			cfg.TensorQuant,
			key,
			w,
			scales,
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

	return nil
}

func collectPerExpertProjection(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, layerPrefix, proj string, numExperts int32) *stackedExpertWeights {
	weights := make([]*mlx.Array, 0, numExperts)
	scales := make([]*mlx.Array, 0, numExperts)
	biases := make([]*mlx.Array, 0, numExperts)
	consumedKeys := make([]string, 0, numExperts*3)
	bits := 0
	groupSize := 0
	mode := cfg.QuantMode

	for e := int32(0); e < numExperts; e++ {
		base := fmt.Sprintf("%s.mlp.experts.%d.%s", layerPrefix, e, proj)
		w, key := tensorByBase(tensors, base)
		if w == nil {
			continue
		}
		consumedKeys = append(consumedKeys, key)

		s := tensors[key+"_scale"]
		if s == nil {
			weights = append(weights, w)
			continue
		}
		consumedKeys = append(consumedKeys, key+"_scale")
		qb := tensors[key+"_qbias"]
		if qb != nil {
			consumedKeys = append(consumedKeys, key+"_qbias")
		}
		gs, b, m := model.ResolveLinearQuantParams(
			cfg.QuantGroupSize,
			cfg.QuantBits,
			cfg.QuantMode,
			cfg.TensorQuant,
			key,
			w,
			s,
		)
		if bits == 0 {
			bits = b
			groupSize = gs
			mode = m
		}
		if useQuantized && supportsGatherQMM(m, b) {
			weights = append(weights, w)
			scales = append(scales, s)
			if qb != nil {
				biases = append(biases, qb)
			}
		} else {
			weights = append(weights, mlx.Dequantize(w, s, qb, gs, b, m))
		}
	}

	if len(weights) == 0 {
		return nil
	}

	out := &stackedExpertWeights{Weight: stackAndDetach(weights), Bits: bits, GroupSize: groupSize, Mode: mode}
	if len(scales) == len(weights) {
		out.Scales = stackAndDetach(scales)
	}
	if len(biases) == len(weights) {
		out.Biases = stackAndDetach(biases)
	}
	freeTensorKeys(tensors, consumedKeys...)
	return out
}

func splitGateUpProjection(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, layerPrefix string) (gate, up, down *stackedExpertWeights) {
	gateUp, key := tensorAny(
		tensors,
		layerPrefix+".mlp.experts.gate_up_proj.weight",
		layerPrefix+".mlp.experts.gate_up_proj",
	)
	if gateUp == nil {
		return nil, nil, nil
	}

	if scales := tensors[key+"_scale"]; scales != nil {
		qbiases := tensors[key+"_qbias"]
		groupSize, bits, mode := model.ResolveLinearQuantParams(
			cfg.QuantGroupSize,
			cfg.QuantBits,
			cfg.QuantMode,
			cfg.TensorQuant,
			key,
			gateUp,
			scales,
		)
		if useQuantized && supportsGatherQMM(mode, bits) {
			gate = &stackedExpertWeights{
				Bits:      bits,
				GroupSize: groupSize,
				Mode:      mode,
			}
			up = &stackedExpertWeights{
				Bits:      bits,
				GroupSize: groupSize,
				Mode:      mode,
			}
			// Keep quantized packed tensor and split along the out-dim (axis=1).
			// This assumes MLX quantization preserves the leading [experts, out, ...] layout.
			if gateUp.NumDims() != 3 {
				return nil, nil, nil
			}
			shape := gateUp.Dims()
			nExperts, twoHidden, inHidden := int32(shape[0]), int32(shape[1]), int32(shape[2])
			_ = nExperts
			_ = inHidden
			mid := twoHidden / 2

			gate.Weight = sliceStackedExpertAxis1(gateUp, 0, mid)
			up.Weight = sliceStackedExpertAxis1(gateUp, mid, twoHidden)
			gate.Scales = sliceStackedExpertAxis1(scales, 0, mid)
			up.Scales = sliceStackedExpertAxis1(scales, mid, twoHidden)
			if qbiases != nil {
				gate.Biases = sliceStackedExpertAxis1(qbiases, 0, mid)
				up.Biases = sliceStackedExpertAxis1(qbiases, mid, twoHidden)
			}
		} else {
			gateUp = mlx.Dequantize(gateUp, scales, qbiases, groupSize, bits, mode)
			gate = &stackedExpertWeights{Bits: bits, GroupSize: groupSize, Mode: mode}
			up = &stackedExpertWeights{Bits: bits, GroupSize: groupSize, Mode: mode}
		}
	}

	if gateUp.NumDims() != 3 {
		return nil, nil, nil
	}
	shape := gateUp.Dims()
	nExperts, twoHidden, inHidden := int32(shape[0]), int32(shape[1]), int32(shape[2])
	mid := twoHidden / 2

	if gate == nil {
		gate = &stackedExpertWeights{}
	}
	if up == nil {
		up = &stackedExpertWeights{}
	}
	if gate.Weight == nil {
		gate.Weight = mlx.SliceStartStop(gateUp, []int32{0, 0, 0}, []int32{nExperts, mid, inHidden})
	}
	if up.Weight == nil {
		up.Weight = mlx.SliceStartStop(gateUp, []int32{0, mid, 0}, []int32{nExperts, twoHidden, inHidden})
	}

	downW, downKey := tensorAny(
		tensors,
		layerPrefix+".mlp.experts.down_proj.weight",
		layerPrefix+".mlp.experts.down_proj",
	)
	if downW == nil {
		return gate, up, nil
	}
	if scales := tensors[downKey+"_scale"]; scales != nil {
		qbiases := tensors[downKey+"_qbias"]
		groupSize, bits, mode := model.ResolveLinearQuantParams(
			cfg.QuantGroupSize,
			cfg.QuantBits,
			cfg.QuantMode,
			cfg.TensorQuant,
			downKey,
			downW,
			scales,
		)
		if useQuantized && supportsGatherQMM(mode, bits) {
			down = &stackedExpertWeights{
				Weight:    downW,
				Scales:    scales,
				Biases:    qbiases,
				Bits:      bits,
				GroupSize: groupSize,
				Mode:      mode,
			}
			return gate, up, down
		}
		downW = mlx.Dequantize(downW, scales, qbiases, groupSize, bits, mode)
		down = &stackedExpertWeights{Bits: bits, GroupSize: groupSize, Mode: mode}
	}
	if down == nil {
		down = &stackedExpertWeights{}
	}
	down.Weight = downW
	return gate, up, down
}

func sanitizeConvWeight(w *mlx.Array) *mlx.Array {
	if w == nil {
		return nil
	}
	if w.NumDims() == 3 {
		if w.Dim(1) == 1 {
			return mlx.Squeeze(w, 1)
		}
		if w.Dim(2) == 1 {
			return mlx.Squeeze(w, 2)
		}
	}
	return w
}

func depthwiseConv1dKernelWeight(w *mlx.Array) *mlx.Array {
	if w == nil {
		return nil
	}
	switch w.NumDims() {
	case 2:
		// qwen3.5 manual path stores [C, K]; MLX grouped conv expects [Cout, K, Cin/groups].
		// For depthwise conv (groups=C), that is [C, K, 1].
		return mlx.ExpandDims(w, 2)
	case 3:
		switch {
		case w.Dim(2) == 1:
			// [C, K, 1]
			return w
		case w.Dim(1) == 1:
			// [C, 1, K] -> [C, K, 1]
			return mlx.Transpose(w, 0, 2, 1)
		case w.Dim(0) == 1:
			// [1, K, C] -> [C, K, 1]
			return mlx.Transpose(w, 2, 1, 0)
		}
	}
	return nil
}

func shouldShiftNormKey(key string) bool {
	for _, suffix := range []string{
		".input_layernorm.weight",
		".post_attention_layernorm.weight",
		"model.norm.weight",
		".self_attn.q_norm.weight",
		".self_attn.k_norm.weight",
	} {
		if strings.HasSuffix(key, suffix) {
			return true
		}
	}
	return false
}

func maybeShiftNormWeight(key string, w *mlx.Array, shouldShift bool) *mlx.Array {
	if !shouldShift || w == nil || w.NumDims() != 1 || !shouldShiftNormKey(key) {
		return w
	}
	return mlx.AddScalar(w, 1.0)
}

// LoadWeights assigns tensors to model fields.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	layout := resolveTensorPathLayout(tensors)
	m.weightPrefix = layout.containerPrefix
	prefix := m.weightPrefix
	modelPrefix := layout.containerPrefix + layout.modelPrefix
	cfg := m.Config

	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)

	shouldShiftNormWeights := false
	mtpKeys := make([]string, 0)
	for name, t := range tensors {
		if strings.Contains(name, "mtp.") {
			shouldShiftNormWeights = true
			mtpKeys = append(mtpKeys, name)
			continue
		}
		if !shouldShiftNormWeights && strings.Contains(name, ".linear_attn.conv1d.weight") && t != nil && t.NumDims() == 3 && t.Dim(2) != 1 {
			shouldShiftNormWeights = true
		}
	}
	if len(mtpKeys) > 0 {
		freeTensorKeys(tensors, mtpKeys...)
	}

	embedKey := modelPrefix + "embed_tokens.weight"
	embedWeight := tensors[embedKey]
	if embedWeight == nil {
		return fmt.Errorf("missing embedding weight: %sembed_tokens.weight", modelPrefix)
	}
	m.EmbedTokens = nn.NewEmbedding(embedWeight)

	normKey := modelPrefix + "norm.weight"
	normWeight := maybeShiftNormWeight(normKey, tensors[normKey], shouldShiftNormWeights)
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: %snorm.weight", modelPrefix)
	}
	m.Norm = nn.NewRMSNorm(normWeight, cfg.RMSNormEps)

	if cfg.TieWordEmbeddings {
		m.LMHead = nn.NewLinear(embedWeight, nil)
	} else if lmHead := linears.Make(prefix + "lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else if lmHead := linears.Make("lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else {
		m.LMHead = nn.NewLinear(embedWeight, nil)
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
	moeLoadSummaries := make([]string, 0)

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		layerPrefix := fmt.Sprintf("%slayers.%d", modelPrefix, i)
		layer := &Layer{IsLinear: layerIsLinear(cfg, i)}

		if w := maybeShiftNormWeight(layerPrefix+".input_layernorm.weight", tensors[layerPrefix+".input_layernorm.weight"], shouldShiftNormWeights); w != nil {
			layer.InputNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if w := maybeShiftNormWeight(layerPrefix+".post_attention_layernorm.weight", tensors[layerPrefix+".post_attention_layernorm.weight"], shouldShiftNormWeights); w != nil {
			layer.PostAttentionNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if layer.InputNorm == nil || layer.PostAttentionNorm == nil {
			return fmt.Errorf("layer %d: missing layer norms", i)
		}

		if layer.IsLinear {
			lin := &GatedDeltaNet{}
			lin.InProjQKV = linears.Make(layerPrefix + ".linear_attn.in_proj_qkv")
			lin.InProjZ = linears.Make(layerPrefix + ".linear_attn.in_proj_z")
			lin.InProjB = linears.Make(layerPrefix + ".linear_attn.in_proj_b")
			lin.InProjA = linears.Make(layerPrefix + ".linear_attn.in_proj_a")
			lin.InProjQKVZ = linears.Make(layerPrefix + ".linear_attn.in_proj_qkvz")
			lin.InProjBA = linears.Make(layerPrefix + ".linear_attn.in_proj_ba")
			lin.OutProj = linears.Make(layerPrefix + ".linear_attn.out_proj")

			lin.ConvWeight = sanitizeConvWeight(tensors[layerPrefix+".linear_attn.conv1d.weight"])
			if lin.ConvWeight == nil {
				lin.ConvWeight = sanitizeConvWeight(tensors[layerPrefix+".linear_attn.conv1d"])
			}
			lin.NormWeight, _ = tensorAny(tensors,
				layerPrefix+".linear_attn.norm.weight",
				layerPrefix+".linear_attn.norm",
			)
			lin.DtBias, _ = tensorAny(tensors,
				layerPrefix+".linear_attn.dt_bias",
				layerPrefix+".linear_attn.dt_proj",
			)
			lin.ALog, _ = tensorAny(tensors,
				layerPrefix+".linear_attn.A_log",
				layerPrefix+".linear_attn.a_log",
			)
			if lin.ALog != nil {
				lin.AExp = mlx.Exp(lin.ALog.AsType(mlx.DTypeFloat32))
			}

			hasSplit := lin.InProjQKV != nil && lin.InProjZ != nil && lin.InProjB != nil && lin.InProjA != nil
			hasCombined := lin.InProjQKVZ != nil && lin.InProjBA != nil
			if (!hasSplit && !hasCombined) || lin.OutProj == nil {
				return fmt.Errorf("layer %d: missing linear attention projections", i)
			}
			if lin.ConvWeight == nil || lin.NormWeight == nil || lin.DtBias == nil || lin.ALog == nil || lin.AExp == nil {
				return fmt.Errorf("layer %d: missing linear attention state tensors", i)
			}
			if lin.ConvWeight.NumDims() != 2 {
				return fmt.Errorf("layer %d: conv1d weight must be 2D after sanitization, got %dD", i, lin.ConvWeight.NumDims())
			}
			if convKernel := depthwiseConv1dKernelWeight(lin.ConvWeight); convKernel != nil {
				lin.Conv1D = nn.NewConv1d(convKernel, nil, 1, 0, 1, int32(lin.ConvWeight.Dim(0)))
			}

			layer.Linear = lin
		} else {
			attn := &FullAttention{}
			attn.QProj = linears.Make(layerPrefix + ".self_attn.q_proj")
			attn.KProj = linears.Make(layerPrefix + ".self_attn.k_proj")
			attn.VProj = linears.Make(layerPrefix + ".self_attn.v_proj")
			attn.OProj = linears.Make(layerPrefix + ".self_attn.o_proj")

			if w := maybeShiftNormWeight(layerPrefix+".self_attn.q_norm.weight", tensors[layerPrefix+".self_attn.q_norm.weight"], shouldShiftNormWeights); w != nil {
				attn.QNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
			}
			if w := maybeShiftNormWeight(layerPrefix+".self_attn.k_norm.weight", tensors[layerPrefix+".self_attn.k_norm.weight"], shouldShiftNormWeights); w != nil {
				attn.KNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
			}

			if attn.QProj == nil || attn.KProj == nil || attn.VProj == nil || attn.OProj == nil {
				return fmt.Errorf("layer %d: missing full attention projections", i)
			}
			if attn.QNorm == nil || attn.KNorm == nil {
				return fmt.Errorf("layer %d: missing full attention q/k norms", i)
			}
			layer.FullAttn = attn
		}

		if layerUsesMoE(cfg, i) {
			moe := &SparseMoE{}
			moe.Gate = linears.Make(layerPrefix + ".mlp.gate")
			if moe.Gate == nil {
				return fmt.Errorf("layer %d: missing moe gate", i)
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
				g2, u2, d2 := splitGateUpProjection(tensors, cfg, useQuantizedExperts, layerPrefix)
				if gateW == nil {
					gateW = g2
				}
				if upW == nil {
					upW = u2
				}
				if downW == nil {
					downW = d2
				}
			}
			if gateW == nil || upW == nil || downW == nil {
				gateW = collectPerExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "gate_proj", cfg.NumExperts)
				upW = collectPerExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "up_proj", cfg.NumExperts)
				downW = collectPerExpertProjection(tensors, cfg, useQuantizedExperts, layerPrefix, "down_proj", cfg.NumExperts)
			}

			if gateW == nil || upW == nil || downW == nil {
				return fmt.Errorf("layer %d: missing switch expert weights", i)
			}

			switchMLP := &SwitchMLP{}
			if gateW.Scales != nil && upW.Scales != nil && downW.Scales != nil {
				switchMLP.UseQuantized = true
				switchMLP.GateWeightQ = gateW.Weight
				switchMLP.GateScales = gateW.Scales
				switchMLP.GateBiases = gateW.Biases
				switchMLP.GateBits = gateW.Bits
				switchMLP.GateGroupSize = gateW.GroupSize
				switchMLP.UpWeightQ = upW.Weight
				switchMLP.UpScales = upW.Scales
				switchMLP.UpBiases = upW.Biases
				switchMLP.UpBits = upW.Bits
				switchMLP.UpGroupSize = upW.GroupSize
				switchMLP.DownWeightQ = downW.Weight
				switchMLP.DownScales = downW.Scales
				switchMLP.DownBiases = downW.Biases
				switchMLP.DownBits = downW.Bits
				switchMLP.DownGroupSize = downW.GroupSize
			} else {
				switchMLP.GateWeight = transposeExpertWeightForGatherMM(gateW.Weight)
				switchMLP.UpWeight = transposeExpertWeightForGatherMM(upW.Weight)
				switchMLP.DownWeight = transposeExpertWeightForGatherMM(downW.Weight)
				moeLoadSummaries = append(moeLoadSummaries,
					fmt.Sprintf(
						"layer=%d moe_fallback reason=%s %s %s %s",
						i,
						summarizeMoEFallbackReason(gateW, upW, downW),
						describeMoEProjection("gate", gateW),
						describeMoEProjection("up", upW),
						describeMoEProjection("down", downW),
					),
				)
			}
			if switchMLP.UseQuantized {
				moeLoadSummaries = append(moeLoadSummaries,
					fmt.Sprintf(
						"layer=%d moe_quantized %s %s %s",
						i,
						describeMoEProjection("gate", gateW),
						describeMoEProjection("up", upW),
						describeMoEProjection("down", downW),
					),
				)
			}
			moe.SwitchMLP = switchMLP

			sharedGateProj := linears.Make(layerPrefix + ".mlp.shared_expert.gate_proj")
			sharedUpProj := linears.Make(layerPrefix + ".mlp.shared_expert.up_proj")
			sharedDownProj := linears.Make(layerPrefix + ".mlp.shared_expert.down_proj")
			if sharedGateProj != nil && sharedUpProj != nil && sharedDownProj != nil {
				moe.SharedExpert = &DenseMLP{
					GateProj: sharedGateProj,
					UpProj:   sharedUpProj,
					DownProj: sharedDownProj,
				}
				moe.SharedExpertGate = linears.Make(layerPrefix + ".mlp.shared_expert_gate")
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

func softplus(x *mlx.Array) *mlx.Array {
	return mlx.Log(mlx.AddScalar(mlx.Exp(x), 1.0))
}

func repeatHeads(x *mlx.Array, repeatFactor int32) *mlx.Array {
	if repeatFactor <= 1 {
		return x
	}
	shape := x.Dims()
	x = mlx.ExpandDims(x, 3)
	x = mlx.Tile(x, []int32{1, 1, 1, repeatFactor, 1})
	return mlx.Reshape(x, int32(shape[0]), int32(shape[1]), int32(shape[2])*repeatFactor, int32(shape[3]))
}

func depthwiseCausalConv1d(x, w *mlx.Array, outLen int32) *mlx.Array {
	if x == nil || w == nil {
		return nil
	}
	if w.NumDims() != 2 {
		return nil
	}
	B := int32(x.Dim(0))
	C := int32(w.Dim(0))
	K := int32(w.Dim(1))
	var out *mlx.Array
	for i := int32(0); i < K; i++ {
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

func splitQKVZBA(mixedQKVZ, mixedBA *mlx.Array, cfg *Config, B, L int32) (q, k, v, z, b, a *mlx.Array) {
	nk := cfg.LinearNumKeyHeads
	nv := cfg.LinearNumValueHeads
	dk := cfg.LinearKeyHeadDim
	dv := cfg.LinearValueHeadDim
	vPerK := nv / nk

	mixedQKVZ = mlx.Reshape(mixedQKVZ, B, L, nk, 2*dk+2*vPerK*dv)
	q = mlx.SliceStartStop(mixedQKVZ, []int32{0, 0, 0, 0}, []int32{B, L, nk, dk})
	k = mlx.SliceStartStop(mixedQKVZ, []int32{0, 0, 0, dk}, []int32{B, L, nk, 2 * dk})
	v = mlx.SliceStartStop(mixedQKVZ, []int32{0, 0, 0, 2 * dk}, []int32{B, L, nk, 2*dk + vPerK*dv})
	z = mlx.SliceStartStop(mixedQKVZ, []int32{0, 0, 0, 2*dk + vPerK*dv}, []int32{B, L, nk, 2*dk + 2*vPerK*dv})

	v = mlx.Reshape(v, B, L, nv, dv)
	z = mlx.Reshape(z, B, L, nv, dv)

	mixedBA = mlx.Reshape(mixedBA, B, L, nk, 2*vPerK)
	b = mlx.SliceStartStop(mixedBA, []int32{0, 0, 0, 0}, []int32{B, L, nk, vPerK})
	a = mlx.SliceStartStop(mixedBA, []int32{0, 0, 0, vPerK}, []int32{B, L, nk, 2 * vPerK})
	b = mlx.Reshape(b, B, L, nv)
	a = mlx.Reshape(a, B, L, nv)

	return q, k, v, z, b, a
}

func (a *FullAttention) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	qg := a.QProj.Forward(x)
	qg = mlx.Reshape(qg, B, L, cfg.NumAttentionHeads, cfg.HeadDim*2)
	q := mlx.SliceStartStop(qg, []int32{0, 0, 0, 0}, []int32{B, L, cfg.NumAttentionHeads, cfg.HeadDim})
	gate := mlx.SliceStartStop(qg, []int32{0, 0, 0, cfg.HeadDim}, []int32{B, L, cfg.NumAttentionHeads, cfg.HeadDim * 2})
	gate = mlx.Reshape(gate, B, L, cfg.NumAttentionHeads*cfg.HeadDim)

	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)
	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)

	q = a.QNorm.Forward(q, cfg.RMSNormEps)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	q = mlx.RoPEWithBase(q, int(cfg.RopeDim), false, cfg.RopeTheta, 1.0, offset)
	k = mlx.RoPEWithBase(k, int(cfg.RopeDim), false, cfg.RopeTheta, 1.0, offset)

	if c != nil {
		k, v = c.Update(k, v)
	}

	out := mlx.ScaledDotProductAttentionCausal(q, k, v, cfg.Scale, L > 1)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	out = mlx.Mul(out, mlx.Sigmoid(gate))
	out = a.OProj.Forward(out)
	return out
}

func (g *GatedDeltaNet) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	var qkv, z, b, a *mlx.Array
	useSplitProj := g.InProjQKV != nil && g.InProjZ != nil && g.InProjB != nil && g.InProjA != nil
	if useSplitProj {
		qkv = g.InProjQKV.Forward(x)
		z = g.InProjZ.Forward(x)
		z = mlx.Reshape(z, B, L, cfg.LinearNumValueHeads, cfg.LinearValueHeadDim)
		b = g.InProjB.Forward(x)
		a = g.InProjA.Forward(x)
	} else {
		mixedQKVZ := g.InProjQKVZ.Forward(x)
		mixedBA := g.InProjBA.Forward(x)
		var q, k, v *mlx.Array
		q, k, v, z, b, a = splitQKVZBA(mixedQKVZ, mixedBA, cfg, B, L)
		qkv = mlx.Concatenate([]*mlx.Array{
			mlx.Reshape(q, B, L, cfg.LinearNumKeyHeads*cfg.LinearKeyHeadDim),
			mlx.Reshape(k, B, L, cfg.LinearNumKeyHeads*cfg.LinearKeyHeadDim),
			mlx.Reshape(v, B, L, cfg.LinearNumValueHeads*cfg.LinearValueHeadDim),
		}, -1)
	}

	convTail := cfg.LinearConvKernelDim - 1
	var convState *mlx.Array
	var rc *cache.RecurrentCache
	if c != nil {
		if typed, ok := c.(*cache.RecurrentCache); ok {
			rc = typed
			convState = rc.ConvState(int(B), x.DType())
		}
	}
	if convState == nil {
		convState = mlx.Zeros(x.DType(), int(B), int(convTail), int(2*cfg.LinearNumKeyHeads*cfg.LinearKeyHeadDim+cfg.LinearNumValueHeads*cfg.LinearValueHeadDim))
	}

	convInput := mlx.Concatenate([]*mlx.Array{convState, qkv}, 1)
	fastRecurrentWrite := L == 1
	var convOut *mlx.Array
	if g.Conv1D != nil {
		convOut = g.Conv1D.Forward(convInput)
	} else {
		convOut = depthwiseCausalConv1d(convInput, g.ConvWeight, L)
	}
	convOut = mlx.SiLU(convOut)
	if rc != nil {
		total := int32(convInput.Dim(1))
		start := total - convTail
		nextConv := mlx.SliceStartStop(convInput, []int32{0, start, 0}, []int32{B, total, int32(convInput.Dim(2))})
		// Always snapshot/materialize recurrent state so prefill does not retain
		// full token-unrolled graphs across requests.
		if fastRecurrentWrite {
			rc.SetConvStateFast(nextConv)
		} else {
			rc.SetConvState(nextConv)
		}
	}

	keyDim := cfg.LinearNumKeyHeads * cfg.LinearKeyHeadDim
	valueDim := cfg.LinearNumValueHeads * cfg.LinearValueHeadDim
	q := mlx.SliceStartStop(convOut, []int32{0, 0, 0}, []int32{B, L, keyDim})
	k := mlx.SliceStartStop(convOut, []int32{0, 0, keyDim}, []int32{B, L, 2 * keyDim})
	v := mlx.SliceStartStop(convOut, []int32{0, 0, 2 * keyDim}, []int32{B, L, 2*keyDim + valueDim})
	q = mlx.Reshape(q, B, L, cfg.LinearNumKeyHeads, cfg.LinearKeyHeadDim)
	k = mlx.Reshape(k, B, L, cfg.LinearNumKeyHeads, cfg.LinearKeyHeadDim)
	v = mlx.Reshape(v, B, L, cfg.LinearNumValueHeads, cfg.LinearValueHeadDim)
	invScale := float32(1.0 / math.Sqrt(float64(cfg.LinearKeyHeadDim)))
	repeatFactor := cfg.LinearNumValueHeads / cfg.LinearNumKeyHeads
	q = mlx.MulScalar(mlx.RMSNormFn(q, nil, 1e-6), invScale*invScale)
	k = mlx.MulScalar(mlx.RMSNormFn(k, nil, 1e-6), invScale)

	aF32 := a.AsType(mlx.DTypeFloat32)
	dtBiasF32 := g.DtBias.AsType(mlx.DTypeFloat32)
	gDecay := softplus(mlx.Add(aF32, dtBiasF32))
	gDecay = mlx.Mul(gDecay, g.AExp)
	gDecay = mlx.Exp(mlx.MulScalar(gDecay, -1))
	gDecay = gDecay.AsType(a.DType())

	beta := mlx.Sigmoid(b)

	var state *mlx.Array
	if rc != nil {
		state = rc.DeltaState(int(B), x.DType())
	}
	if state == nil {
		state = mlx.Zeros(x.DType(), int(B), int(cfg.LinearNumValueHeads), int(cfg.LinearValueHeadDim), int(cfg.LinearKeyHeadDim))
	}

	var out *mlx.Array
	if fusedOut, fusedState, fused := mlx.GatedDeltaKernel(q, k, v, gDecay, beta, state); fused {
		out = fusedOut
		state = fusedState
	} else if L == 1 {
		if repeatFactor > 1 {
			q = repeatHeads(q, repeatFactor)
			k = repeatHeads(k, repeatFactor)
		}
		// Fast decode path: avoid per-token slice/append graph construction.
		qt := mlx.Squeeze(q, 1)
		kt := mlx.Squeeze(k, 1)
		vt := mlx.Squeeze(v, 1)
		gt := mlx.Squeeze(gDecay, 1)
		bt := mlx.Squeeze(beta, 1)

		state = mlx.Mul(state, mlx.ExpandDims(mlx.ExpandDims(gt, -1), -1))
		kvMem := mlx.Sum(mlx.Mul(state, mlx.ExpandDims(kt, 2)), -1, false)
		delta := mlx.Mul(mlx.Sub(vt, kvMem), mlx.ExpandDims(bt, -1))
		state = mlx.Add(state, mlx.Mul(mlx.ExpandDims(kt, 2), mlx.ExpandDims(delta, -1)))
		yt := mlx.Sum(mlx.Mul(state, mlx.ExpandDims(qt, 2)), -1, false)
		out = mlx.ExpandDims(yt, 1)
	} else {
		if repeatFactor > 1 {
			q = repeatHeads(q, repeatFactor)
			k = repeatHeads(k, repeatFactor)
		}
		outs := make([]*mlx.Array, 0, L)
		for t := int32(0); t < L; t++ {
			qt := mlx.Squeeze(mlx.SliceStartStop(q, []int32{0, t, 0, 0}, []int32{B, t + 1, cfg.LinearNumValueHeads, cfg.LinearKeyHeadDim}), 1)
			kt := mlx.Squeeze(mlx.SliceStartStop(k, []int32{0, t, 0, 0}, []int32{B, t + 1, cfg.LinearNumValueHeads, cfg.LinearKeyHeadDim}), 1)
			vt := mlx.Squeeze(mlx.SliceStartStop(v, []int32{0, t, 0, 0}, []int32{B, t + 1, cfg.LinearNumValueHeads, cfg.LinearValueHeadDim}), 1)
			gt := mlx.Squeeze(mlx.SliceStartStop(gDecay, []int32{0, t, 0}, []int32{B, t + 1, cfg.LinearNumValueHeads}), 1)
			bt := mlx.Squeeze(mlx.SliceStartStop(beta, []int32{0, t, 0}, []int32{B, t + 1, cfg.LinearNumValueHeads}), 1)

			state = mlx.Mul(state, mlx.ExpandDims(mlx.ExpandDims(gt, -1), -1))
			kvMem := mlx.Sum(mlx.Mul(state, mlx.ExpandDims(kt, 2)), -1, false)
			delta := mlx.Mul(mlx.Sub(vt, kvMem), mlx.ExpandDims(bt, -1))
			state = mlx.Add(state, mlx.Mul(mlx.ExpandDims(kt, 2), mlx.ExpandDims(delta, -1)))
			yt := mlx.Sum(mlx.Mul(state, mlx.ExpandDims(qt, 2)), -1, false)
			outs = append(outs, mlx.ExpandDims(yt, 1))
		}
		out = mlx.Concatenate(outs, 1)
	}
	out = mlx.RMSNormFn(out, g.NormWeight, cfg.RMSNormEps)
	out = mlx.Mul(out, mlx.SiLU(z))
	out = mlx.Reshape(out, B, L, valueDim)
	out = g.OutProj.Forward(out)
	if rc != nil {
		if fastRecurrentWrite {
			rc.SetDeltaStateFast(state)
		} else {
			rc.SetDeltaState(state)
		}
		rc.Advance(int(L))
	}
	return out
}

func (m *DenseMLP) Forward(x *mlx.Array, _ *Config) *mlx.Array {
	return m.DownProj.Forward(mlx.Mul(mlx.SiLU(m.GateProj.Forward(x)), m.UpProj.Forward(x)))
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
		gate = mlx.GatherQMM(xFlat, s.GateWeightQ, s.GateScales, s.GateBiases,
			nil, idxFlat, true, s.GateGroupSize, s.GateBits, cfg.QuantMode, doSort)
		up = mlx.GatherQMM(xFlat, s.UpWeightQ, s.UpScales, s.UpBiases,
			nil, idxFlat, true, s.UpGroupSize, s.UpBits, cfg.QuantMode, doSort)
		hidden = mlx.Mul(mlx.SiLU(gate), up)
		down = mlx.GatherQMM(hidden, s.DownWeightQ, s.DownScales, s.DownBiases,
			nil, idxFlat, true, s.DownGroupSize, s.DownBits, cfg.QuantMode, doSort)
	} else {
		gate = mlx.GatherMM(xFlat, s.GateWeight, nil, idxFlat, doSort)
		up = mlx.GatherMM(xFlat, s.UpWeight, nil, idxFlat, doSort)
		hidden = mlx.Mul(mlx.SiLU(gate), up)
		down = mlx.GatherMM(hidden, s.DownWeight, nil, idxFlat, doSort)
	}

	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

func (m *SparseMoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	probs := mlx.SoftmaxAxis(m.Gate.Forward(x), -1, true)
	neg := mlx.Neg(probs)
	inds := mlx.Argpartition(neg, int(cfg.NumExpertsPerTok)-1, -1)
	shape := inds.Dims()
	inds = mlx.SliceStartStop(inds, []int32{0, 0, 0}, []int32{int32(shape[0]), int32(shape[1]), cfg.NumExpertsPerTok})

	scores := mlx.TakeAlongAxis(probs, inds, -1)
	if cfg.NormTopKProb && cfg.NumExpertsPerTok > 1 {
		sumScores := mlx.Sum(scores, -1, true)
		scores = mlx.Div(scores, sumScores)
	}

	expertOut := m.SwitchMLP.Forward(x, inds, cfg)
	y := mlx.Sum(mlx.Mul(expertOut, mlx.ExpandDims(scores, -1)), 2, false)

	if m.SharedExpert != nil {
		shared := m.SharedExpert.Forward(x, cfg)
		if m.SharedExpertGate != nil {
			shared = mlx.Mul(shared, mlx.Sigmoid(m.SharedExpertGate.Forward(x)))
		}
		y = mlx.Add(y, shared)
	}

	return mlx.Reshape(y, B, L, cfg.HiddenSize)
}

func (l *Layer) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	var r *mlx.Array
	normed := l.InputNorm.Forward(x, cfg.RMSNormEps)
	if l.IsLinear {
		r = l.Linear.Forward(normed, c, B, L, cfg)
	} else {
		r = l.FullAttn.Forward(normed, c, B, L, cfg)
	}
	h := mlx.Add(x, r)
	r = l.MLP.Forward(l.PostAttentionNorm.Forward(h, cfg.RMSNormEps), cfg)
	return mlx.Add(h, r)
}

func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	dims := tokens.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	h := m.EmbedTokens.Forward(tokens)
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, c, B, L, m.Config)
	}
	out := m.Norm.Forward(h, m.RMSNormEps)
	return out
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

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	convTail := m.LinearConvKernelDim - 1
	convDim := 2*m.LinearNumKeyHeads*m.LinearKeyHeadDim + m.LinearNumValueHeads*m.LinearValueHeadDim
	for i, layer := range m.Layers {
		if layer.IsLinear {
			caches[i] = cache.NewRecurrentCache(convTail, convDim, m.LinearNumValueHeads, m.LinearValueHeadDim, m.LinearKeyHeadDim)
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

// DisablePromptCache returns false to allow append-only prompt cache reuse.
// Recurrent caches report CanTrim=false, so divergent prefixes are dropped.
func (m *Model) DisablePromptCache() bool {
	return false
}
