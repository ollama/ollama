// Package granitemoe provides a GraniteMoe-style sparse mixture-of-experts
// decoder-only transformer for MLX.
//
// GraniteMoe is dense Granite (see x/models/granite) with every layer's SwiGLU
// MLP replaced by a sparse mixture of experts, plus the same four Llama-style
// scalar multipliers (embeddings, attention, residual, logits).
package granitemoe

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
	base.Register("GraniteMoeForCausalLM", newModel)
}

// Config holds GraniteMoe model configuration.
type Config struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`

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

	// Quantization parameters (set during load based on model quantization).
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	HeadDim int32   `json:"-"`
	Scale   float32 `json:"-"`
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

func resolveWeightPrefix(tensors map[string]*mlx.Array) string {
	for _, prefix := range []string{"", "language_model."} {
		if tensors[prefix+"model.embed_tokens.weight"] != nil {
			return prefix
		}
	}
	return ""
}

func newModel(root *model.Root) (base.Model, error) {
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
		cfg.AttentionMultiplier = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
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
	if w == nil || !w.Valid() || w.NumDims() != 3 {
		return w
	}
	t := mlx.Transpose(w, 0, 2, 1)
	cloned := t.Clone()
	mlx.Eval(cloned)
	return cloned
}

// splitAxisHalves splits a into two equal halves along axis.
func splitAxisHalves(a *mlx.Array, axis int32) (lo, hi *mlx.Array) {
	dims := a.Dims()
	nd := len(dims)
	beg := make([]int32, nd)
	end := make([]int32, nd)
	for i, d := range dims {
		end[i] = int32(d)
	}
	mid := int32(dims[axis]) / 2
	endLo := append([]int32(nil), end...)
	endLo[axis] = mid
	begHi := append([]int32(nil), beg...)
	begHi[axis] = mid
	return mlx.SliceStartStop(a, beg, endLo), mlx.SliceStartStop(a, begHi, end)
}

// splitFusedGateUp splits a fused [gate; up] stacked expert projection
// (shape [E, 2*intermediate, hidden], concatenated along the output axis —
// the layout HF's native input_linear tensor uses) into its two halves,
// slicing any quantization scale/bias companions identically since affine
// group quantization indexes them per output row.
func splitFusedGateUp(fused *stackedExpertWeights) (lo, hi *stackedExpertWeights) {
	loW, hiW := splitAxisHalves(fused.Weight, 1)
	lo = &stackedExpertWeights{Weight: loW, Bits: fused.Bits, GroupSize: fused.GroupSize, Mode: fused.Mode}
	hi = &stackedExpertWeights{Weight: hiW, Bits: fused.Bits, GroupSize: fused.GroupSize, Mode: fused.Mode}
	if fused.Scales != nil {
		lo.Scales, hi.Scales = splitAxisHalves(fused.Scales, 1)
	}
	if fused.Biases != nil {
		lo.Biases, hi.Biases = splitAxisHalves(fused.Biases, 1)
	}
	return lo, hi
}

// loadStackedProjection loads an already-stacked (single tensor covering all
// experts) projection by its exact checkpoint name.
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
		Weight:    mlx.Dequantize(w, scales, qbiases, groupSize, bits, mode, nil),
		Bits:      bits,
		GroupSize: groupSize,
		Mode:      mode,
	}
}

// loadExpertProjections resolves a layer's gate/up/down expert projections.
// It prefers the separately-stacked switch_mlp layout that mlx_lm's own
// GraniteMoe conversion ships (block_sparse_moe.switch_mlp.{gate,up,down}_proj),
// and falls back to the fused layout HF checkpoints ship natively
// (input_linear packs [gate; up] along the output axis; output_linear is
// unchanged).
func loadExpertProjections(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, moePrefix string) (gate, up, down *stackedExpertWeights) {
	gate = loadStackedProjection(tensors, cfg, useQuantized, moePrefix+".switch_mlp.gate_proj")
	up = loadStackedProjection(tensors, cfg, useQuantized, moePrefix+".switch_mlp.up_proj")
	down = loadStackedProjection(tensors, cfg, useQuantized, moePrefix+".switch_mlp.down_proj")

	if down == nil {
		down = loadStackedProjection(tensors, cfg, useQuantized, moePrefix+".output_linear")
	}
	if gate == nil || up == nil {
		if fused := loadStackedProjection(tensors, cfg, useQuantized, moePrefix+".input_linear"); fused != nil {
			gate, up = splitFusedGateUp(fused)
		}
	}
	return gate, up, down
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

		gateW, upW, downW := loadExpertProjections(tensors, cfg, useQuantizedExperts, layerPrefix+".block_sparse_moe")
		if gateW == nil || upW == nil || downW == nil {
			return fmt.Errorf("layer %d: missing moe expert weights", i)
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

	q = mlx.RoPEWithBase(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions)
	k = mlx.RoPEWithBase(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions)

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

	var gate, up, down *mlx.Array
	if s.UseQuantized {
		gate = mlx.GatherQMM(xFlat, s.GateWeightQ, s.GateScales, s.GateBiases,
			nil, idxFlat, true, s.GateGroupSize, s.GateBits, s.GateMode, doSort)
		up = mlx.GatherQMM(xFlat, s.UpWeightQ, s.UpScales, s.UpBiases,
			nil, idxFlat, true, s.UpGroupSize, s.UpBits, s.UpMode, doSort)
	} else {
		gate = mlx.GatherMM(xFlat, s.GateWeight, nil, idxFlat, doSort)
		up = mlx.GatherMM(xFlat, s.UpWeight, nil, idxFlat, doSort)
	}
	hidden := mlx.SwiGLU(gate, up)

	if s.UseQuantized {
		down = mlx.GatherQMM(hidden, s.DownWeightQ, s.DownScales, s.DownBiases,
			nil, idxFlat, true, s.DownGroupSize, s.DownBits, s.DownMode, doSort)
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
