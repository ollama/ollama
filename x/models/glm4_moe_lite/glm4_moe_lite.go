//go:build mlx

// Package glm4_moe_lite provides the GLM4-MoE-Lite implementation for MLX.
// This model uses Multi-head Latent Attention (MLA) and Mixture of Experts (MoE).
package glm4_moe_lite

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
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
type MLAAttention struct {
	QAProj      nn.LinearLayer
	QALayerNorm *nn.RMSNorm
	QBProj      nn.LinearLayer

	KVAProjWithMQA nn.LinearLayer
	KVALayerNorm   *nn.RMSNorm

	EmbedQ     *nn.MultiLinear
	UnembedOut *nn.MultiLinear

	OProj nn.LinearLayer
}

// Forward computes absorbed MLA attention output.
func (a *MLAAttention) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	q := a.QAProj.Forward(x)
	q = a.QALayerNorm.Forward(q, cfg.RMSNormEps)
	q = a.QBProj.Forward(q)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.QHeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)

	qNope := mlx.SliceStartStop(q, []int32{0, 0, 0, 0}, []int32{B, cfg.NumAttentionHeads, L, cfg.QKNopeHeadDim})
	qPE := mlx.SliceStartStop(q, []int32{0, 0, 0, cfg.QKNopeHeadDim}, []int32{B, cfg.NumAttentionHeads, L, cfg.QHeadDim})

	compressedKV := a.KVAProjWithMQA.Forward(x)

	kvCompressed := mlx.SliceStartStop(compressedKV, []int32{0, 0, 0}, []int32{B, L, cfg.KVLoraRank})
	kPE := mlx.SliceStartStop(compressedKV, []int32{0, 0, cfg.KVLoraRank}, []int32{B, L, cfg.KVLoraRank + cfg.QKRopeHeadDim})

	kPE = mlx.Reshape(kPE, B, L, 1, cfg.QKRopeHeadDim)
	kPE = mlx.Transpose(kPE, 0, 2, 1, 3)

	kvLatent := a.KVALayerNorm.Forward(kvCompressed, cfg.RMSNormEps)
	kvLatent = mlx.ExpandDims(kvLatent, 1)

	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	qPE = mlx.RoPEWithBase(qPE, int(cfg.QKRopeHeadDim), true, cfg.RopeTheta, 1.0, offset)
	kPE = mlx.RoPEWithBase(kPE, int(cfg.QKRopeHeadDim), true, cfg.RopeTheta, 1.0, offset)

	qLatent := a.EmbedQ.Forward(qNope)

	keys := mlx.Concatenate([]*mlx.Array{kvLatent, kPE}, 3)

	cachedL := L
	if c != nil {
		placeholderValues := mlx.ZerosF32([]int32{B, 1, L, 0})
		keys, _ = c.Update(keys, placeholderValues)
		cachedL = int32(keys.Dim(2))
	}

	values := mlx.SliceStartStop(keys, []int32{0, 0, 0, 0}, []int32{B, 1, cachedL, cfg.KVLoraRank})

	queries := mlx.Concatenate([]*mlx.Array{qLatent, qPE}, 3)

	out := mlx.ScaledDotProductAttentionCausal(queries, keys, values, cfg.Scale, L > 1)

	out = a.UnembedOut.Forward(out)

	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.VHeadDim)

	return a.OProj.Forward(out)
}

// DenseMLP implements the standard SwiGLU MLP for dense layers
type DenseMLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

// Forward applies the SwiGLU MLP
func (m *DenseMLP) Forward(x *mlx.Array) *mlx.Array {
	gate := mlx.SiLU(m.GateProj.Forward(x))
	up := m.UpProj.Forward(x)
	return m.DownProj.Forward(mlx.Mul(gate, up))
}

// MoEGate implements the expert gating mechanism
type MoEGate struct {
	Gate                 nn.LinearLayer
	EScoreCorrectionBias *mlx.Array
}

// Forward computes expert selection indices and scores
func (g *MoEGate) Forward(x *mlx.Array, cfg *Config) (*mlx.Array, *mlx.Array) {
	gates := g.Gate.Forward(x)

	scores := mlx.Sigmoid(gates)
	origScores := scores

	if g.EScoreCorrectionBias != nil {
		scores = mlx.Add(scores, g.EScoreCorrectionBias)
	}

	topK := cfg.NumExpertsPerTok
	negScores := mlx.Neg(scores)
	inds := mlx.Argpartition(negScores, int(topK)-1, -1)

	dims := inds.Dims()
	inds = mlx.SliceStartStop(inds, []int32{0, 0, 0}, []int32{int32(dims[0]), int32(dims[1]), topK})

	scores = mlx.TakeAlongAxis(origScores, inds, -1)

	if topK > 1 && cfg.NormTopKProb {
		sumScores := mlx.Sum(scores, -1, true)
		scores = mlx.Div(scores, sumScores)
	}

	scores = mlx.MulScalar(scores, cfg.RoutedScalingFactor)

	return inds, scores
}

// SwitchMLP implements the MoE expert computation using stacked weights
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

// Forward applies the switched expert MLP
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
		gate = mlx.GatherMM(xFlat, mlx.Transpose(s.GateWeight, 0, 2, 1), nil, idxFlat, doSort)
		up = mlx.GatherMM(xFlat, mlx.Transpose(s.UpWeight, 0, 2, 1), nil, idxFlat, doSort)

		hidden = mlx.Mul(mlx.SiLU(gate), up)

		down = mlx.GatherMM(hidden, mlx.Transpose(s.DownWeight, 0, 2, 1), nil, idxFlat, doSort)
	}

	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

// SharedExperts implements the shared expert MLP
type SharedExperts struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
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
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	inds, scores := m.Gate.Forward(x, cfg)

	expertOut := m.SwitchMLP.Forward(x, inds, cfg)

	scoresExpanded := mlx.ExpandDims(scores, -1)
	y := mlx.Sum(mlx.Mul(expertOut, scoresExpanded), 2, false)

	if m.SharedExperts != nil {
		y = mlx.Add(y, m.SharedExperts.Forward(x))
	}

	return mlx.Reshape(y, B, L, cfg.HiddenSize)
}

// DenseBlock represents a dense transformer block (for first_k_dense_replace layers)
type DenseBlock struct {
	Attention              *MLAAttention
	MLP                    *DenseMLP
	InputLayerNorm         *nn.RMSNorm
	PostAttentionLayerNorm *nn.RMSNorm
}

// Forward applies the dense block
func (b *DenseBlock) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	r := b.Attention.Forward(b.InputLayerNorm.Forward(x, cfg.RMSNormEps), c, B, L, cfg)
	h := mlx.Add(x, r)

	r = b.MLP.Forward(b.PostAttentionLayerNorm.Forward(h, cfg.RMSNormEps))
	return mlx.Add(h, r)
}

// MoEBlock represents a MoE transformer block
type MoEBlock struct {
	Attention              *MLAAttention
	MoE                    *MoE
	InputLayerNorm         *nn.RMSNorm
	PostAttentionLayerNorm *nn.RMSNorm
}

// Forward applies the MoE block
func (b *MoEBlock) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	r := b.Attention.Forward(b.InputLayerNorm.Forward(x, cfg.RMSNormEps), c, B, L, cfg)
	h := mlx.Add(x, r)

	r = b.MoE.Forward(b.PostAttentionLayerNorm.Forward(h, cfg.RMSNormEps), cfg)
	return mlx.Add(h, r)
}

// Block interface for both dense and MoE blocks
type Block interface {
	Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array
}

// Model represents the complete GLM4-MoE-Lite model
type Model struct {
	EmbedTokens *nn.Embedding
	Layers      []Block
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config
}

// computeScale computes the attention scale.
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
func supportsGatherQMM(mode string, bits int) bool {
	return mode == "affine" && (bits == 4 || bits == 8)
}

// quantizationParams returns groupSize, bits, mode for a quantization type string.
func quantizationParams(quantization string) (groupSize, bits int, mode string) {
	switch strings.ToUpper(quantization) {
	case "NVFP4":
		return 16, 4, "nvfp4"
	case "FP4", "Q4", "INT4":
		return 32, 4, "affine"
	case "MXFP8":
		return 32, 8, "mxfp8"
	case "FP8", "Q8", "INT8", "":
		return 64, 8, "affine"
	default:
		return 32, 8, "affine"
	}
}

// readBlobMetadata reads the __metadata__ from a safetensors blob header.
func readBlobMetadata(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, err
	}
	if headerSize > 1024*1024 {
		return nil, fmt.Errorf("header too large: %d", headerSize)
	}

	data := make([]byte, headerSize)
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, err
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, err
	}

	metaRaw, ok := header["__metadata__"]
	if !ok {
		return nil, nil
	}

	var meta map[string]string
	if err := json.Unmarshal(metaRaw, &meta); err != nil {
		return nil, err
	}
	return meta, nil
}

// ExpertWeight holds a single expert's weight with optional quantization components.
type ExpertWeight struct {
	Weight    *mlx.Array
	Scales    *mlx.Array
	Biases    *mlx.Array
	Bits      int
	GroupSize int
}

// loadExpertWeight loads an expert weight from the tensor map.
func loadExpertWeight(tensors map[string]*mlx.Array, path string, useQuantized bool, cfg *Config) *ExpertWeight {
	w := tensors[path+".weight"]
	if w == nil {
		return nil
	}

	scales := tensors[path+".weight_scale"]
	if scales != nil {
		qbiases := tensors[path+".weight_qbias"]

		groupSize, bits, mode := cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode

		if useQuantized && supportsGatherQMM(mode, bits) {
			return &ExpertWeight{Weight: w, Scales: scales, Biases: qbiases, Bits: bits, GroupSize: groupSize}
		}

		return &ExpertWeight{Weight: mlx.Dequantize(w, scales, qbiases, groupSize, bits, mode)}
	}

	return &ExpertWeight{Weight: w}
}

// StackedExpertWeights holds stacked weights for all experts.
type StackedExpertWeights struct {
	Weight    *mlx.Array
	Scales    *mlx.Array
	Biases    *mlx.Array
	Bits      int
	GroupSize int
}

// collectAndStackExpertWeights loads and stacks expert weights for one projection type.
func collectAndStackExpertWeights(
	tensors map[string]*mlx.Array,
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
		ew := loadExpertWeight(tensors, path, useQuantized, cfg)
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
func sanitizeExpertWeights(tensors map[string]*mlx.Array, prefix string, numExperts int32, useQuantized bool, cfg *Config) (gate, up, down *StackedExpertWeights) {
	gate = collectAndStackExpertWeights(tensors, prefix, "gate_proj", numExperts, useQuantized, cfg)
	up = collectAndStackExpertWeights(tensors, prefix, "up_proj", numExperts, useQuantized, cfg)
	down = collectAndStackExpertWeights(tensors, prefix, "down_proj", numExperts, useQuantized, cfg)
	return gate, up, down
}

// sanitizeMLAWeights transforms kv_b_proj weights into absorbed MLA format.
func sanitizeMLAWeights(tensors map[string]*mlx.Array, prefix string, cfg *Config) (*mlx.Array, *mlx.Array) {
	path := prefix + ".self_attn.kv_b_proj"
	w := tensors[path+".weight"]
	if w == nil {
		return nil, nil
	}

	// Check if quantized and dequantize
	if scales := tensors[path+".weight_scale"]; scales != nil {
		qbiases := tensors[path+".weight_qbias"]
		w = mlx.Dequantize(w, scales, qbiases, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode)
	}

	headDim := cfg.QKNopeHeadDim + cfg.VHeadDim
	w = mlx.Reshape(w, cfg.NumAttentionHeads, headDim, cfg.KVLoraRank)

	wk := mlx.SliceStartStop(w, []int32{0, 0, 0}, []int32{cfg.NumAttentionHeads, cfg.QKNopeHeadDim, cfg.KVLoraRank})
	wv := mlx.SliceStartStop(w, []int32{0, cfg.QKNopeHeadDim, 0}, []int32{cfg.NumAttentionHeads, headDim, cfg.KVLoraRank})

	embedQ := mlx.Transpose(wk, 0, 2, 1)
	unembedOut := wv

	return embedQ, unembedOut
}

// makeLinear creates a Linear or QuantizedLinear layer from the tensor map.
func makeLinear(tensors map[string]*mlx.Array, path string, cfg *Config) nn.LinearLayer {
	w := tensors[path+".weight"]
	if w == nil {
		return nil
	}

	scales := tensors[path+".weight_scale"]
	if scales != nil {
		qbiases := tensors[path+".weight_qbias"]
		bias := tensors[path+".bias"]
		return &nn.QuantizedLinear{
			Weight:    w,
			Scales:    scales,
			QBiases:   qbiases,
			Bias:      bias,
			GroupSize: cfg.QuantGroupSize,
			Bits:      cfg.QuantBits,
			Mode:      cfg.QuantMode,
		}
	}

	bias := tensors[path+".bias"]
	return nn.NewLinear(w, bias)
}

// LoadFromManifest loads a GLM4-MoE-Lite model from a manifest (Ollama blob storage).
func LoadFromManifest(modelManifest *manifest.ModelManifest) (*Model, error) {
	configData, err := modelManifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	cfg.QHeadDim = cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	cfg.Scale = computeScale(&cfg)

	// Load all tensors from manifest blobs into a flat map
	allTensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool) // dedupe by digest
	var quantType string
	var quantGroupSize int

	for _, layer := range modelManifest.GetTensorLayers("") {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		blobPath := modelManifest.BlobPath(layer.Digest)

		// Read quantization metadata from first blob
		if quantType == "" {
			if meta, err := readBlobMetadata(blobPath); err == nil && meta != nil {
				if qt := meta["quant_type"]; qt != "" {
					quantType = strings.ToUpper(qt)
				}
				if gs := meta["group_size"]; gs != "" {
					fmt.Sscanf(gs, "%d", &quantGroupSize)
				}
			}
		}

		for name, arr := range mlx.Load(blobPath) {
			// Map safetensors key naming to our naming convention
			// Combined blobs use ".scale" and ".bias" suffixes
			if strings.HasSuffix(name, ".scale") {
				baseName := strings.TrimSuffix(name, ".scale")
				allTensors[baseName+"_scale"] = arr
			} else if strings.HasSuffix(name, ".bias") && !strings.HasSuffix(name, ".weight_qbias") {
				// Check if this is a quantization bias or a regular bias
				// by checking if there's a corresponding weight
				baseName := strings.TrimSuffix(name, ".bias")
				if _, hasScale := allTensors[baseName+"_scale"]; hasScale {
					allTensors[baseName+"_qbias"] = arr
				} else {
					allTensors[name] = arr
				}
			} else {
				allTensors[name] = arr
			}
		}
	}

	// Set up quantization parameters
	useQuantized := false
	if quantType != "" {
		_, cfg.QuantBits, cfg.QuantMode = quantizationParams(quantType)
		if quantGroupSize > 0 {
			cfg.QuantGroupSize = quantGroupSize
		} else {
			cfg.QuantGroupSize, _, _ = quantizationParams(quantType)
		}
		useQuantized = supportsGatherQMM(cfg.QuantMode, cfg.QuantBits)
	}

	// Load tokenizer
	tokData, err := modelManifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer config: %w", err)
	}

	tokConfig := &tokenizer.TokenizerConfig{
		ConfigJSON: configData,
	}

	if genConfigData, err := modelManifest.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = genConfigData
	}

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

	// Load embedding
	if w := allTensors["model.embed_tokens.weight"]; w != nil {
		m.EmbedTokens = nn.NewEmbedding(w)
	}

	// Load final norm
	if w := allTensors["model.norm.weight"]; w != nil {
		m.Norm = nn.NewRMSNorm(w, cfg.RMSNormEps)
	}

	// Load LM head
	m.LMHead = makeLinear(allTensors, "lm_head", &cfg)

	// Load layers
	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)

		// Load attention (same for both block types)
		attn := &MLAAttention{}
		attn.QAProj = makeLinear(allTensors, prefix+".self_attn.q_a_proj", &cfg)
		if w := allTensors[prefix+".self_attn.q_a_layernorm.weight"]; w != nil {
			attn.QALayerNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		attn.QBProj = makeLinear(allTensors, prefix+".self_attn.q_b_proj", &cfg)
		attn.KVAProjWithMQA = makeLinear(allTensors, prefix+".self_attn.kv_a_proj_with_mqa", &cfg)
		if w := allTensors[prefix+".self_attn.kv_a_layernorm.weight"]; w != nil {
			attn.KVALayerNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		attn.OProj = makeLinear(allTensors, prefix+".self_attn.o_proj", &cfg)

		// Sanitize MLA weights for absorbed attention
		embedQ, unembedOut := sanitizeMLAWeights(allTensors, prefix, &cfg)
		attn.EmbedQ = nn.NewMultiLinear(embedQ)
		attn.UnembedOut = nn.NewMultiLinear(unembedOut)

		inputLN := allTensors[prefix+".input_layernorm.weight"]
		postAttnLN := allTensors[prefix+".post_attention_layernorm.weight"]

		if i < cfg.FirstKDenseReplace {
			// Dense block
			block := &DenseBlock{Attention: attn}
			if inputLN != nil {
				block.InputLayerNorm = nn.NewRMSNorm(inputLN, cfg.RMSNormEps)
			}
			if postAttnLN != nil {
				block.PostAttentionLayerNorm = nn.NewRMSNorm(postAttnLN, cfg.RMSNormEps)
			}

			block.MLP = &DenseMLP{
				GateProj: makeLinear(allTensors, prefix+".mlp.gate_proj", &cfg),
				UpProj:   makeLinear(allTensors, prefix+".mlp.up_proj", &cfg),
				DownProj: makeLinear(allTensors, prefix+".mlp.down_proj", &cfg),
			}

			m.Layers[i] = block
		} else {
			// MoE block
			block := &MoEBlock{Attention: attn}
			if inputLN != nil {
				block.InputLayerNorm = nn.NewRMSNorm(inputLN, cfg.RMSNormEps)
			}
			if postAttnLN != nil {
				block.PostAttentionLayerNorm = nn.NewRMSNorm(postAttnLN, cfg.RMSNormEps)
			}

			// Stack expert weights
			gate, up, down := sanitizeExpertWeights(allTensors, prefix, cfg.NRoutedExperts, useQuantized, &cfg)

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

			moeGate := &MoEGate{}
			moeGate.Gate = makeLinear(allTensors, prefix+".mlp.gate", &cfg)
			if bias := allTensors[prefix+".mlp.gate.e_score_correction_bias"]; bias != nil {
				moeGate.EScoreCorrectionBias = bias
			}

			block.MoE = &MoE{
				Gate:      moeGate,
				SwitchMLP: switchMLP,
			}

			// Load shared experts if present
			if cfg.NSharedExperts > 0 {
				block.MoE.SharedExperts = &SharedExperts{
					GateProj: makeLinear(allTensors, prefix+".mlp.shared_experts.gate_proj", &cfg),
					UpProj:   makeLinear(allTensors, prefix+".mlp.shared_experts.up_proj", &cfg),
					DownProj: makeLinear(allTensors, prefix+".mlp.shared_experts.down_proj", &cfg),
				}
			}

			m.Layers[i] = block
		}
	}

	mlx.Eval(mlx.Collect(m)...)

	return m, nil
}

// Forward computes the forward pass of the model
func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	dims := tokens.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	h := m.EmbedTokens.Forward(tokens)

	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil {
			c = caches[i]
		}
		h = layer.Forward(h, c, B, L, m.Config)
	}

	h = m.Norm.Forward(h, m.RMSNormEps)
	return h
}

// Unembed applies the LM head to get logits.
func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(x)
}

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
func (m *Model) FormatPrompt(prompt string) string {
	return "[gMASK]<sop><|user|>" + prompt + "<|assistant|><think>"
}

// FormatPromptWithThinking applies the GLM-4 chat template with explicit thinking control.
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
