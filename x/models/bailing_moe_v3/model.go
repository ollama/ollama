// Package bailing_moe_v3 implements the Bailing MoE V3 hybrid KDA/MLA model.
package bailing_moe_v3

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

// mlaLayerEval commits and waits for the op stream after each MLA layer
// during prefill (OLLAMA_MLA_LAYER_EVAL=1 enables). Superseded by the
// blocked prefill path, which bounds transients by construction.
var mlaLayerEval = os.Getenv("OLLAMA_MLA_LAYER_EVAL") == "1"

// mlaChunkedPrefill processes the latent history in fixed-size key blocks
// with an online softmax during prefill (OLLAMA_MLA_CHUNKED_PREFILL=0
// falls back to re-expanding the whole history per chunk).
var mlaChunkedPrefill = os.Getenv("OLLAMA_MLA_CHUNKED_PREFILL") != "0"

func init() {
	base.Register("BailingMoeV3ForCausalLM", NewModel)
}

var _ base.Model = (*Model)(nil)

// Config is the subset of BailingMoeV3Config needed for inference.
type Config struct {
	ModelType             string          `json:"model_type"`
	HiddenSize            int32           `json:"hidden_size"`
	IntermediateSize      int32           `json:"intermediate_size"`
	NumHiddenLayers       int32           `json:"num_hidden_layers"`
	NumAttentionHeads     int32           `json:"num_attention_heads"`
	NumKeyValueHeads      int32           `json:"num_key_value_heads"`
	HeadDim               int32           `json:"head_dim"`
	VocabSize             int32           `json:"vocab_size"`
	MaxPositionEmbeddings int32           `json:"max_position_embeddings"`
	RMSNormEps            float32         `json:"rms_norm_eps"`
	RopeTheta             float32         `json:"rope_theta"`
	RopeScaling           json.RawMessage `json:"rope_scaling"`
	TieWordEmbeddings     bool            `json:"tie_word_embeddings"`

	LayerGroupSize       int32   `json:"layer_group_size"`
	ShortConvKernelSize  int32   `json:"short_conv_kernel_size"`
	NoKDALora            bool    `json:"no_kda_lora"`
	KDASafeGate          bool    `json:"kda_safe_gate"`
	KDALowerBound        float32 `json:"kda_lower_bound"`
	GatedAttentionType   string  `json:"gated_attention_proj_granularity_type"`
	QKHeadDim            int32   `json:"qk_head_dim"`
	QKNopeHeadDim        int32   `json:"qk_nope_head_dim"`
	QKRopeHeadDim        int32   `json:"qk_rope_head_dim"`
	VHeadDim             int32   `json:"v_head_dim"`
	QLoraRank            int32   `json:"q_lora_rank"`
	KVLoraRank           int32   `json:"kv_lora_rank"`
	RopeInterleave       bool    `json:"rope_interleave"`
	UseQKNorm            bool    `json:"use_qk_norm"`
	FirstKDenseReplace   int32   `json:"first_k_dense_replace"`
	NumExperts           int32   `json:"num_experts"`
	NumExpertsPerTok     int32   `json:"num_experts_per_tok"`
	NumSharedExperts     int32   `json:"num_shared_experts"`
	MoEIntermediateSize  int32   `json:"moe_intermediate_size"`
	SharedIntermediate   int32   `json:"moe_shared_expert_intermediate_size"`
	NGroup               int32   `json:"n_group"`
	TopKGroup            int32   `json:"topk_group"`
	NormTopKProb         bool    `json:"norm_topk_prob"`
	RoutedScalingFactor  float32 `json:"routed_scaling_factor"`
	RouterUsesExpertBias bool    `json:"moe_router_enable_expert_bias"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	Scale float32 `json:"-"`
}

type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config
}

type Attention interface {
	Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array
}

type MLP interface {
	Forward(x *mlx.Array, cfg *Config) *mlx.Array
}

type Layer struct {
	InputNorm         *nn.RMSNorm
	PostAttentionNorm *nn.RMSNorm
	Attention         Attention
	MLP               MLP
	IsMLA             bool
}

type DenseMLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer

	// GateUpProj, when non-nil, is the row-fused (GateProj, UpProj) pair:
	// one wide matmul whose left half is the gate and right half the up.
	GateUpProj nn.LinearLayer
}

func (m *DenseMLP) Forward(x *mlx.Array, _ *Config) *mlx.Array {
	if m.GateUpProj != nil {
		gateUp := m.GateUpProj.Forward(x)
		dims := gateUp.Dims()
		B, L := int32(dims[0]), int32(dims[1])
		half := int32(dims[2]) / 2
		return m.DownProj.Forward(mlx.SwiGLU(
			sliceCols(gateUp, B, L, 0, half),
			sliceCols(gateUp, B, L, half, 2*half),
		))
	}
	return m.DownProj.Forward(mlx.SwiGLU(m.GateProj.Forward(x), m.UpProj.Forward(x)))
}

// MLAAttention implements the checkpoint's multi-latent attention with
// DeepSeek-style absorption: the KV cache stores only the 576-dim compressed
// latent (kv_lora_rank + rope) per token instead of the expanded per-head
// keys/values (num_heads * (qk_head_dim + v_head_dim) = 5120 dims), an 8.9x
// KV memory saving. kv_b_proj's two halves are folded into the query path
// (EmbedQ) and the output path (UnembedOut); the attention core runs as MQA
// over the shared latent. Mathematically equivalent to the expanded form.
type MLAAttention struct {
	QProj          nn.LinearLayer
	QAProj         nn.LinearLayer
	QALayerNorm    *nn.RMSNorm
	QBProj         nn.LinearLayer
	KVAProjWithMQA nn.LinearLayer
	KVALayerNorm   *nn.RMSNorm
	EmbedQ         *nn.MultiLinear // [H, kv_lora_rank, qk_nope_head_dim]: absorbs kv_b_proj's K half into Q
	UnembedOut     *nn.MultiLinear // [H, v_head_dim, kv_lora_rank]: absorbs kv_b_proj's V half into the output
	ExpandKW       *mlx.Array      // [kv_lora_rank, H*qk_nope_head_dim]: re-expands latent history to keys as one 2D GEMM
	ExpandVW       *mlx.Array      // [kv_lora_rank, H*v_head_dim]: re-expands latent history to values as one 2D GEMM
	GateProj       nn.LinearLayer
	OutProj        nn.LinearLayer
}

func parseConfig(data []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}
	if raw := bytes.TrimSpace(cfg.RopeScaling); len(raw) > 0 && !bytes.Equal(raw, []byte("null")) {
		return Config{}, fmt.Errorf("rope_scaling is not supported")
	}

	switch {
	case cfg.HiddenSize <= 0:
		return Config{}, fmt.Errorf("invalid hidden_size: %d", cfg.HiddenSize)
	case cfg.NumHiddenLayers <= 0:
		return Config{}, fmt.Errorf("invalid num_hidden_layers: %d", cfg.NumHiddenLayers)
	case cfg.NumAttentionHeads <= 0:
		return Config{}, fmt.Errorf("invalid num_attention_heads: %d", cfg.NumAttentionHeads)
	case cfg.HeadDim <= 0:
		return Config{}, fmt.Errorf("invalid head_dim: %d", cfg.HeadDim)
	case cfg.LayerGroupSize <= 0:
		return Config{}, fmt.Errorf("invalid layer_group_size: %d", cfg.LayerGroupSize)
	case !cfg.NoKDALora:
		return Config{}, fmt.Errorf("KDA LoRA projections are not supported yet")
	case cfg.QKNopeHeadDim+cfg.QKRopeHeadDim != cfg.QKHeadDim:
		return Config{}, fmt.Errorf("qk dimensions do not close: %d + %d != %d", cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.QKHeadDim)
	}

	if cfg.NumExperts > 0 {
		switch {
		case cfg.NGroup <= 0:
			return Config{}, fmt.Errorf("invalid n_group: %d", cfg.NGroup)
		case cfg.NumExperts%cfg.NGroup != 0:
			return Config{}, fmt.Errorf("num_experts (%d) must be divisible by n_group (%d)", cfg.NumExperts, cfg.NGroup)
		case cfg.NumExperts/cfg.NGroup < 2:
			return Config{}, fmt.Errorf("experts per group must be at least 2: num_experts=%d n_group=%d", cfg.NumExperts, cfg.NGroup)
		case cfg.TopKGroup <= 0:
			return Config{}, fmt.Errorf("invalid topk_group: %d", cfg.TopKGroup)
		case cfg.TopKGroup > cfg.NGroup:
			return Config{}, fmt.Errorf("topk_group (%d) must not exceed n_group (%d)", cfg.TopKGroup, cfg.NGroup)
		case cfg.NumExpertsPerTok <= 0:
			return Config{}, fmt.Errorf("invalid num_experts_per_tok: %d", cfg.NumExpertsPerTok)
		case cfg.NumExpertsPerTok > cfg.TopKGroup*(cfg.NumExperts/cfg.NGroup):
			return Config{}, fmt.Errorf("num_experts_per_tok (%d) exceeds the %d candidates selected by topk_group", cfg.NumExpertsPerTok, cfg.TopKGroup*(cfg.NumExperts/cfg.NGroup))
		}
	}

	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.ShortConvKernelSize <= 0 {
		cfg.ShortConvKernelSize = 4
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	if cfg.RoutedScalingFactor == 0 {
		cfg.RoutedScalingFactor = 1
	}
	if cfg.SharedIntermediate == 0 {
		cfg.SharedIntermediate = cfg.MoEIntermediateSize * cfg.NumSharedExperts
	}
	if !cfg.RopeInterleave {
		return Config{}, fmt.Errorf("only interleaved MLA RoPE is supported")
	}
	if cfg.GatedAttentionType != "head_wise" {
		return Config{}, fmt.Errorf("unsupported gated attention type %q", cfg.GatedAttentionType)
	}

	cfg.Scale = float32(1 / math.Sqrt(float64(cfg.QKHeadDim)))
	return cfg, nil
}

func isMLALayer(cfg *Config, layer int32) bool {
	return (layer+1)%cfg.LayerGroupSize == 0 ||
		layer >= cfg.NumHiddenLayers/cfg.LayerGroupSize*cfg.LayerGroupSize
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
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	tokCfg := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	if data, err := root.Manifest.ReadConfig("generation_config.json"); err == nil {
		tokCfg.GenerationConfigJSON = data
	}
	if data, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokCfg.TokenizerConfigJSON = data
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokCfg)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	return &Model{
		Layers: make([]*Layer, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	cfg := m.Config
	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)

	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, "model.word_embeddings", cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	if m.EmbedTokens == nil {
		// Keep the conventional Hugging Face spelling as a compatibility alias.
		m.EmbedTokens = model.MakeEmbeddingLayer(tensors, "model.embed_tokens", cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	}
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing model.word_embeddings.weight")
	}
	if w := tensors["model.norm.weight"]; w != nil {
		m.Norm = nn.NewRMSNorm(w, cfg.RMSNormEps)
	}
	if m.Norm == nil {
		return fmt.Errorf("missing model.norm.weight")
	}
	if cfg.TieWordEmbeddings {
		m.LMHead = m.EmbedTokens.AsLinear()
	} else {
		m.LMHead = linears.Make("lm_head")
	}
	if m.LMHead == nil {
		return fmt.Errorf("missing lm_head.weight")
	}

	for i := range cfg.NumHiddenLayers {
		prefix := fmt.Sprintf("model.layers.%d", i)
		layer := &Layer{IsMLA: isMLALayer(cfg, i)}
		if w := tensors[prefix+".input_layernorm.weight"]; w != nil {
			layer.InputNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if w := tensors[prefix+".post_attention_layernorm.weight"]; w != nil {
			layer.PostAttentionNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
		if layer.InputNorm == nil || layer.PostAttentionNorm == nil {
			return fmt.Errorf("layer %d: missing layer norm", i)
		}

		var err error
		if layer.IsMLA {
			layer.Attention, err = loadMLAAttention(linears, tensors, prefix, cfg)
		} else {
			layer.Attention, err = loadKDAAttention(linears, tensors, prefix, cfg)
		}
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}

		if i < cfg.FirstKDenseReplace {
			dense := &DenseMLP{
				GateProj: linears.Make(prefix + ".mlp.gate_proj"),
				UpProj:   linears.Make(prefix + ".mlp.up_proj"),
				DownProj: linears.Make(prefix + ".mlp.down_proj"),
			}
			if dense.GateProj == nil || dense.UpProj == nil || dense.DownProj == nil {
				return fmt.Errorf("layer %d: missing dense MLP projection", i)
			}
			dense.fuseGateUp()
			layer.MLP = dense
		} else {
			layer.MLP, err = loadSparseMoE(linears, tensors, prefix, cfg)
			if err != nil {
				return fmt.Errorf("layer %d: %w", i, err)
			}
		}
		m.Layers[i] = layer
	}
	return nil
}

func loadMLAAttention(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, cfg *Config) (*MLAAttention, error) {
	p := prefix + ".attention"
	a := &MLAAttention{
		KVAProjWithMQA: linears.Make(p + ".kv_a_proj_with_mqa"),
		GateProj:       linears.Make(p + ".g_proj"),
		OutProj:        linears.Make(p + ".dense"),
	}
	if cfg.QLoraRank > 0 {
		a.QAProj = linears.Make(p + ".q_a_proj")
		a.QBProj = linears.Make(p + ".q_b_proj")
		if w := tensors[p+".q_a_layernorm.weight"]; w != nil {
			a.QALayerNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
		}
	} else {
		a.QProj = linears.Make(p + ".q_proj")
	}
	if w := tensors[p+".kv_a_layernorm.weight"]; w != nil {
		a.KVALayerNorm = nn.NewRMSNorm(w, cfg.RMSNormEps)
	}
	embedQ, unembedOut, expandK, expandV, err := absorbKVBProj(tensors, p+".kv_b_proj", cfg)
	if err != nil {
		return nil, err
	}
	if embedQ != nil {
		a.EmbedQ = nn.NewMultiLinear(embedQ)
		a.UnembedOut = nn.NewMultiLinear(unembedOut)
		a.ExpandKW = expandK
		a.ExpandVW = expandV
	}
	qProjectionOK := a.QProj != nil ||
		(a.QAProj != nil && a.QALayerNorm != nil && a.QBProj != nil)
	if !qProjectionOK || a.KVAProjWithMQA == nil || a.KVALayerNorm == nil || a.EmbedQ == nil ||
		a.GateProj == nil || a.OutProj == nil {
		return nil, fmt.Errorf("missing MLA projection")
	}
	return a, nil
}

// absorbKVBProj splits kv_b_proj [H*(qk_nope+v_head), kv_lora_rank] into the
// absorbed per-head factors: EmbedQ [H, kv_lora_rank, qk_nope] mapping
// queries into latent space, and UnembedOut [H, v_head, kv_lora_rank]
// mapping latent attention output back to per-head values. Quantized weights
// are dequantized first — kv_b_proj is small (~2.4 MB/layer at bf16), so
// holding the absorbed factors unquantized costs almost nothing while the KV
// cache shrinks 8.9x.
func absorbKVBProj(tensors map[string]*mlx.Array, path string, cfg *Config) (embedQ, unembedOut, expandK, expandV *mlx.Array, err error) {
	w := tensors[path+".weight"]
	if w == nil {
		return nil, nil, nil, nil, nil
	}
	if scales := tensors[path+".weight_scale"]; scales != nil {
		qbiases := tensors[path+".weight_qbias"]
		groupSize, bits, mode := model.ResolveLinearQuantParams(
			cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant,
			path+".weight", w, scales,
		)
		w = mlx.Dequantize(w, scales, qbiases, groupSize, bits, mode, nil)
	}

	headDim := cfg.QKNopeHeadDim + cfg.VHeadDim
	if int64(w.Dim(0)) != int64(cfg.NumAttentionHeads)*int64(headDim) || int32(w.Dim(1)) != cfg.KVLoraRank {
		return nil, nil, nil, nil, fmt.Errorf("kv_b_proj shape %v does not match heads=%d qk_nope+v=%d kv_lora_rank=%d",
			w.Dims(), cfg.NumAttentionHeads, headDim, cfg.KVLoraRank)
	}
	w = mlx.Reshape(w, cfg.NumAttentionHeads, headDim, cfg.KVLoraRank)
	wk := mlx.SliceStartStop(w, []int32{0, 0, 0}, []int32{cfg.NumAttentionHeads, cfg.QKNopeHeadDim, cfg.KVLoraRank})
	wv := mlx.SliceStartStop(w, []int32{0, cfg.QKNopeHeadDim, 0}, []int32{cfg.NumAttentionHeads, headDim, cfg.KVLoraRank})

	embedQ = mlx.Contiguous(mlx.Transpose(wk, 0, 2, 1), false)
	unembedOut = mlx.Contiguous(wv, false)
	// [H, dim, lora] -> [lora, H*dim]: one 2D GEMM re-expands the latent
	// history for every head with no broadcasted batch dims (CUDA's batched
	// matmul materializes broadcasts).
	expandK = mlx.Contiguous(mlx.Reshape(
		mlx.Transpose(wk, 2, 0, 1),
		cfg.KVLoraRank, cfg.NumAttentionHeads*cfg.QKNopeHeadDim), false)
	expandV = mlx.Contiguous(mlx.Reshape(
		mlx.Transpose(wv, 2, 0, 1),
		cfg.KVLoraRank, cfg.NumAttentionHeads*cfg.VHeadDim), false)
	mlx.Eval(embedQ, unembedOut, expandK, expandV)
	delete(tensors, path+".weight")
	delete(tensors, path+".weight_scale")
	delete(tensors, path+".weight_qbias")
	return embedQ, unembedOut, expandK, expandV, nil
}

// interleavedToHalf converts adjacent rotary pairs [x0,x1,x2,x3,...]
// into the half-split layout [x0,x2,...,x1,x3,...] used by the checkpoint's
// apply_rotary_pos_emb_interleave implementation.
func interleavedToHalf(x *mlx.Array) *mlx.Array {
	dims := x.Dims()
	d := dims[len(dims)-1]
	dims32 := make([]int32, len(dims))
	for i, dim := range dims {
		dims32[i] = int32(dim)
	}
	paired := append(append([]int32(nil), dims32[:len(dims32)-1]...), int32(d/2), 2)
	x = mlx.Reshape(x, paired...)
	axes := make([]int, len(paired))
	for i := range axes {
		axes[i] = i
	}
	axes[len(axes)-2], axes[len(axes)-1] = axes[len(axes)-1], axes[len(axes)-2]
	x = mlx.Transpose(x, axes...)
	return mlx.Contiguous(mlx.Reshape(x, dims32...), false)
}

func (a *MLAAttention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	var q *mlx.Array
	if a.QProj != nil {
		q = a.QProj.Forward(x)
	} else {
		q = a.QBProj.Forward(a.QALayerNorm.Forward(a.QAProj.Forward(x), cfg.RMSNormEps))
	}
	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.QKHeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	qNope := mlx.SliceStartStop(q,
		[]int32{0, 0, 0, 0},
		[]int32{B, cfg.NumAttentionHeads, L, cfg.QKNopeHeadDim})
	qPE := mlx.SliceStartStop(q,
		[]int32{0, 0, 0, cfg.QKNopeHeadDim},
		[]int32{B, cfg.NumAttentionHeads, L, cfg.QKHeadDim})

	compressed := a.KVAProjWithMQA.Forward(x)
	kvCompressed := mlx.SliceStartStop(compressed,
		[]int32{0, 0, 0},
		[]int32{B, L, cfg.KVLoraRank})
	kPE := mlx.SliceStartStop(compressed,
		[]int32{0, 0, cfg.KVLoraRank},
		[]int32{B, L, cfg.KVLoraRank + cfg.QKRopeHeadDim})
	kPE = mlx.Transpose(mlx.Reshape(kPE, B, L, 1, cfg.QKRopeHeadDim), 0, 2, 1, 3)
	kvLatent := mlx.ExpandDims(a.KVALayerNorm.Forward(kvCompressed, cfg.RMSNormEps), 1)

	qPE = mlx.RoPEWithBase(interleavedToHalf(qPE), int(cfg.QKRopeHeadDim), false, cfg.RopeTheta, 1, positions)
	kPE = mlx.RoPEWithBase(interleavedToHalf(kPE), int(cfg.QKRopeHeadDim), false, cfg.RopeTheta, 1, positions)

	// Absorbed attention: queries move into the shared latent space
	// (q_nope @ EmbedQ), and the cache stores one 576-dim latent row per
	// token — [kvLatent, kPE] as keys, with V the kvLatent prefix of the
	// same tensor. The attention core is MQA over that single KV head;
	// scores match the expanded form exactly because
	// (q W_k^T) . latent == q . (W_k latent).
	keys := mlx.Concatenate([]*mlx.Array{kvLatent, kPE}, 3)

	var out *mlx.Array
	if typed, ok := c.(cache.Attention); ok {
		placeholderValues := mlx.ZerosF32([]int32{B, 1, L, 0})
		history := typed.Update(b, keys, placeholderValues)
		if L == 1 {
			// Decode: absorbed MQA over the 576-dim latent, written as
			// explicit broadcast matmuls. MLX has no fused SDPA for 576-dim
			// heads on CUDA, and its generic fallback materializes the
			// broadcast K per query head (~300 MB per token per layer at 4K
			// context), which OOMs under the async pipeline; the stride-0
			// broadcast matmul reads the latent once instead. B==1 decode:
			// the history view is the valid region, causal is trivial.
			qLatent := a.EmbedQ.Forward(qNope)
			queries := mlx.Concatenate([]*mlx.Array{qLatent, qPE}, 3)
			hk := history.K()
			T := int32(hk.Dim(2))
			latentDim := cfg.KVLoraRank + cfg.QKRopeHeadDim
			// [B*H, latent] x [latent, T]: plain GEMMs, no broadcast batch.
			q2 := mlx.Reshape(queries, B*cfg.NumAttentionHeads, latentDim)
			hk2 := mlx.Reshape(hk, T, latentDim)
			scores := q2.Matmul(mlx.Transpose(hk2, 1, 0))
			scores = mlx.MulScalar(scores.AsType(mlx.DTypeFloat32), cfg.Scale)
			probs := mlx.SoftmaxAxis(scores, -1, true).AsType(hk.DType())
			hv := mlx.SliceStartStop(hk2, []int32{0, 0}, []int32{T, cfg.KVLoraRank})
			ov := probs.Matmul(hv)
			out = mlx.Reshape(ov, B, cfg.NumAttentionHeads, 1, cfg.KVLoraRank)
			out = a.UnembedOut.Forward(out)
		} else if B == 1 && L <= 4 {
			// Short decode windows (speculative verify): the same absorbed
			// MQA with a right-aligned causal mask. The prefill branch below
			// re-expands the whole latent history to per-head keys/values
			// (two [S, lora] x [lora, H*dim] GEMMs plus tile/copies per layer
			// per round, ~1ms/layer at 4K context), which dominated deep-KV
			// verify rounds; the masked latent path reads the history once.
			qLatent := a.EmbedQ.Forward(qNope)
			queries := mlx.Concatenate([]*mlx.Array{qLatent, qPE}, 3)
			hk := history.K()
			S := int32(hk.Dim(2))
			latentDim := cfg.KVLoraRank + cfg.QKRopeHeadDim
			// [B*H, L, latent] x [latent, S]: strided GEMM, no broadcast batch.
			q2 := mlx.Reshape(queries, B*cfg.NumAttentionHeads, L, latentDim)
			hk2 := mlx.Reshape(hk, S, latentDim)
			scores := q2.Matmul(mlx.Transpose(hk2, 1, 0))
			scores = mlx.MulScalar(scores.AsType(mlx.DTypeFloat32), cfg.Scale)
			// Row t attends to absolute positions [0, S-L+t]: keep = lower
			// triangle offset by the history the window sits on top of.
			keep := mlx.Tri(L, S, int(S-L))
			mask := mlx.Where(keep, mlx.FromValue(float32(0)), mlx.FromValue(float32(-1e9)))
			scores = mlx.Add(scores, mask)
			probs := mlx.SoftmaxAxis(scores, -1, true).AsType(hk.DType())
			hv := mlx.SliceStartStop(hk2, []int32{0, 0}, []int32{S, cfg.KVLoraRank})
			ov := probs.Matmul(hv)
			out = mlx.Reshape(ov, B, cfg.NumAttentionHeads, L, cfg.KVLoraRank)
			out = a.UnembedOut.Forward(out)
		} else if mlaChunkedPrefill {
			// Prefill: absorbed MQA over the latent history, processed in
			// fixed-size key blocks with an online softmax. CUDA's fused
			// SDPA cannot run 576-dim latent heads, and both alternatives
			// materialize O(S) tensors per layer — re-expanding the history
			// to per-head keys/values costs S×24 KiB and the generic SDPA
			// fallback materializes [H, L, S] f32 scores — which stacked up
			// to ~0.3 MB/token of transients and OOM'd long-context
			// prefills. Blocking keeps the transient footprint constant
			// regardless of context length.
			qLatent := a.EmbedQ.Forward(qNope)
			queries := mlx.Concatenate([]*mlx.Array{qLatent, qPE}, 3)
			hk := history.K()
			S := int32(hk.Dim(2))
			latentDim := cfg.KVLoraRank + cfg.QKRopeHeadDim
			q2 := mlx.Reshape(queries, B*cfg.NumAttentionHeads, L, latentDim)
			hk2 := mlx.Reshape(hk, S, latentDim)
			histBase := S - L

			const mlaKeyBlock = 8192
			var acc, rowMax, rowSum *mlx.Array // [B*H, L, lora] f32, [B*H, L, 1] f32 ×2
			for j0 := int32(0); j0 < S; j0 += mlaKeyBlock {
				j1 := min(j0+mlaKeyBlock, S)
				kj := mlx.SliceStartStop(hk2, []int32{j0, 0}, []int32{j1, latentDim})
				kjT := mlx.Transpose(kj, 1, 0)
				raw := q2.Matmul(kjT)
				rawF32 := raw.AsType(mlx.DTypeFloat32)
				scores := mlx.MulScalar(rawF32, cfg.Scale)
				var keep, mask, masked *mlx.Array
				if j1 > histBase {
					// Row t attends to absolute positions <= histBase+t;
					// block column s is absolute j0+s.
					keep = mlx.Tri(L, j1-j0, int(histBase-j0))
					mask = mlx.Where(keep, mlx.FromValue(float32(0)), mlx.FromValue(float32(-1e9)))
					masked = mlx.Add(scores, mask)
				} else {
					masked = scores
				}
				bm := masked.MaxAxis(-1, true)
				nm := bm
				if rowMax != nil {
					nm = mlx.Maximum(rowMax, bm)
				}
				shifted := mlx.Sub(masked, nm)
				p := mlx.Exp(shifted)
				pl := p.AsType(hk.DType())
				vj := mlx.SliceStartStop(hk2, []int32{j0, 0}, []int32{j1, cfg.KVLoraRank})
				pvRaw := pl.Matmul(vj)
				pv := pvRaw.AsType(mlx.DTypeFloat32)
				ps := mlx.Sum(p, -1, true)
				prevAcc, prevMax, prevSum := acc, rowMax, rowSum
				if acc == nil {
					acc, rowMax, rowSum = pv, nm, ps
				} else {
					correct := mlx.Exp(mlx.Sub(prevMax, nm))
					acc = mlx.Add(mlx.Mul(prevAcc, correct), pv)
					rowSum = mlx.Add(mlx.Mul(prevSum, correct), ps)
					rowMax = nm
					mlx.Free(correct)
				}
				// Evaluate the block synchronously, then drop every
				// intermediate (each op's output has an explicit variable so
				// nothing is left implicitly referenced): the block's
				// transients recycle before the next block allocates instead
				// of piling up until the runner's end-of-chunk Sweep.
				mlx.Eval(acc, rowMax, rowSum)
				mlx.Free(kj, kjT, raw, rawF32, scores, keep, mask, shifted, p, pl, vj, pvRaw)
				if bm != rowMax {
					mlx.Free(bm)
				}
				if pv != acc {
					mlx.Free(pv)
				}
				if ps != rowSum {
					mlx.Free(ps)
				}
				if prevAcc != acc {
					mlx.Free(prevAcc, prevSum)
				}
				if prevMax != rowMax && prevMax != nil {
					mlx.Free(prevMax)
				}
				if masked != scores {
					mlx.Free(masked)
				}
			}
			ov := mlx.Div(acc, rowSum).AsType(hk.DType())
			out = mlx.Reshape(ov, B, cfg.NumAttentionHeads, L, cfg.KVLoraRank)
			out = a.UnembedOut.Forward(out)
		} else {
			// Prefill: CUDA's fused SDPA only supports head dims <= 128,
			// so re-expand the latent history to per-head 192/128 keys and
			// values (one broadcasted GEMM each) and run the fast kernel.
			hk := history.K()
			T := int32(hk.Dim(2))
			hk2 := mlx.Reshape(hk, T, cfg.KVLoraRank+cfg.QKRopeHeadDim)
			hl2 := mlx.SliceStartStop(hk2, []int32{0, 0}, []int32{T, cfg.KVLoraRank})
			hpe := mlx.SliceStartStop(hk,
				[]int32{0, 0, 0, cfg.KVLoraRank},
				[]int32{B, 1, T, cfg.KVLoraRank + cfg.QKRopeHeadDim})
			// One [T, lora] x [lora, H*dim] GEMM per tensor, then reshape to
			// per-head layout — no broadcasted batch dims anywhere.
			kNope := mlx.ExpandDims(mlx.Transpose(mlx.Reshape(
				hl2.Matmul(a.ExpandKW),
				T, cfg.NumAttentionHeads, cfg.QKNopeHeadDim), 1, 0, 2), 0)
			histValues := mlx.ExpandDims(mlx.Transpose(mlx.Reshape(
				hl2.Matmul(a.ExpandVW),
				T, cfg.NumAttentionHeads, cfg.VHeadDim), 1, 0, 2), 0)
			histKeys := mlx.Concatenate(
				[]*mlx.Array{kNope, mlx.Tile(hpe, []int32{1, cfg.NumAttentionHeads, 1, 1})}, 3)
			queries := mlx.Concatenate([]*mlx.Array{qNope, qPE}, 3)
			kv := nn.WithExpandedHistory(history, histKeys, histValues)
			out = nn.ScaledDotProductAttention(b, queries, cfg.Scale, kv, nn.WithMask(nn.CausalMask()))
		}
	} else {
		qLatent := a.EmbedQ.Forward(qNope)
		queries := mlx.Concatenate([]*mlx.Array{qLatent, qPE}, 3)
		values := mlx.SliceStartStop(keys, []int32{0, 0, 0, 0}, []int32{B, 1, L, cfg.KVLoraRank})
		kv := nn.WithKV(keys, values, b.SeqQueryLens)
		out = nn.ScaledDotProductAttention(b, queries, cfg.Scale, kv, nn.WithMask(nn.CausalMask()))
		out = a.UnembedOut.Forward(out)
	}

	gate := mlx.Sigmoid(a.GateProj.Forward(x).AsType(mlx.DTypeFloat32))
	gate = mlx.ExpandDims(mlx.Transpose(gate, 0, 2, 1), -1)
	out = mlx.Mul(out.AsType(mlx.DTypeFloat32), gate).AsType(x.DType())
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.VHeadDim)
	return a.OutProj.Forward(out)
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	r := l.Attention.Forward(l.InputNorm.Forward(x, cfg.RMSNormEps), b, c, positions, B, L, cfg)
	h := mlx.Add(x, r)
	r = l.MLP.Forward(l.PostAttentionNorm.Forward(h, cfg.RMSNormEps), cfg)
	return mlx.Add(h, r)
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	h := m.EmbedTokens.Forward(b.InputIDs)
	// Optional belt-and-braces: wait out the stream after each MLA layer
	// (off by default; the blocked prefill path above bounds transients).
	splitMLA := mlaLayerEval && L > 4
	for i, layer := range m.Layers {
		var c cache.Cache
		if i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
		if splitMLA && layer.IsMLA {
			mlx.Eval(h)
		}
	}
	out := m.Norm.Forward(h, m.RMSNormEps)
	return out, out
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array { return m.LMHead.Forward(x) }
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }
func (m *Model) MaxContextLength() int           { return int(m.MaxPositionEmbeddings) }

// DefaultPrefillChunk caps prefill chunks at 256 tokens. The hybrid stack's
// per-chunk transients — KDA per-step states plus MoE/MLA intermediates
// across every layer, all held until the end-of-chunk Sweep — scale with
// chunk length and made larger defaults exhaust the memory budgets these
// models deploy on (GB10 121G with flash, M4 Pro 48G with tiny).
// OLLAMA_PREFILL_CHUNK still overrides.
func (m *Model) DefaultPrefillChunk() int { return 256 }

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	convTail := m.ShortConvKernelSize - 1
	convDim := 3 * m.NumAttentionHeads * m.HeadDim
	for i, layer := range m.Layers {
		if layer.IsMLA {
			caches[i] = cache.NewKVCache()
		} else {
			caches[i] = cache.NewRecurrentCache(convTail, convDim, m.NumAttentionHeads, m.HeadDim, m.HeadDim)
		}
	}
	return caches
}
