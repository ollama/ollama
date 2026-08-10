// Package bailing_moe_v3 implements the Bailing MoE V3 hybrid KDA/MLA model.
package bailing_moe_v3

import (
	"bytes"
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
}

func (m *DenseMLP) Forward(x *mlx.Array, _ *Config) *mlx.Array {
	return m.DownProj.Forward(mlx.SwiGLU(m.GateProj.Forward(x), m.UpProj.Forward(x)))
}

// MLAAttention implements the checkpoint's eager multi-latent attention path.
// The KV cache stores the expanded per-head keys and values.
type MLAAttention struct {
	QProj          nn.LinearLayer
	QAProj         nn.LinearLayer
	QALayerNorm    *nn.RMSNorm
	QBProj         nn.LinearLayer
	KVAProjWithMQA nn.LinearLayer
	KVALayerNorm   *nn.RMSNorm
	KVBProj        nn.LinearLayer
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
		KVBProj:        linears.Make(p + ".kv_b_proj"),
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
	qProjectionOK := a.QProj != nil ||
		(a.QAProj != nil && a.QALayerNorm != nil && a.QBProj != nil)
	if !qProjectionOK || a.KVAProjWithMQA == nil || a.KVALayerNorm == nil || a.KVBProj == nil ||
		a.GateProj == nil || a.OutProj == nil {
		return nil, fmt.Errorf("missing MLA projection")
	}
	return a, nil
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
	kvLatent := a.KVALayerNorm.Forward(kvCompressed, cfg.RMSNormEps)
	kvExpanded := a.KVBProj.Forward(kvLatent)
	kvExpanded = mlx.Transpose(mlx.Reshape(
		kvExpanded, B, L, cfg.NumAttentionHeads, cfg.QKNopeHeadDim+cfg.VHeadDim,
	), 0, 2, 1, 3)
	kNope := mlx.SliceStartStop(kvExpanded,
		[]int32{0, 0, 0, 0},
		[]int32{B, cfg.NumAttentionHeads, L, cfg.QKNopeHeadDim})
	values := mlx.SliceStartStop(kvExpanded,
		[]int32{0, 0, 0, cfg.QKNopeHeadDim},
		[]int32{B, cfg.NumAttentionHeads, L, cfg.QKNopeHeadDim + cfg.VHeadDim})

	qPE = mlx.RoPEWithBase(interleavedToHalf(qPE), int(cfg.QKRopeHeadDim), false, cfg.RopeTheta, 1, positions)
	kPE = mlx.RoPEWithBase(interleavedToHalf(kPE), int(cfg.QKRopeHeadDim), false, cfg.RopeTheta, 1, positions)
	kPE = mlx.Tile(kPE, []int32{1, cfg.NumAttentionHeads, 1, 1})
	keys := mlx.Concatenate([]*mlx.Array{kNope, kPE}, 3)

	var kv nn.SDPAOption
	if typed, ok := c.(cache.Attention); ok {
		history := typed.Update(b, keys, values)
		kv = nn.WithKVHistory(history)
	} else {
		kv = nn.WithKV(keys, values, b.SeqQueryLens)
	}

	queries := mlx.Concatenate([]*mlx.Array{qNope, qPE}, 3)
	out := nn.ScaledDotProductAttention(b, queries, cfg.Scale, kv, nn.WithMask(nn.CausalMask()))

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

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	h := m.EmbedTokens.Forward(b.InputIDs)
	for i, layer := range m.Layers {
		var c cache.Cache
		if i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
	}
	return m.Norm.Forward(h, m.RMSNormEps)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array { return m.LMHead.Forward(x) }
func (m *Model) NumLayers() int                  { return len(m.Layers) }
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }
func (m *Model) MaxContextLength() int           { return int(m.MaxPositionEmbeddings) }

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
