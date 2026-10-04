// Package kolibri1 implements Aleph Alpha Kolibri 1 for MLX.
package kolibri1

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() {
	model.Register("Kolibri1ForCausalLM", newModel)
}

// Config holds Kolibri 1 model configuration.
type Config struct {
	HiddenSize                   int32    `json:"hidden_size"`
	NumHiddenLayers              int32    `json:"num_hidden_layers"`
	NumExperts                   int32    `json:"num_experts"`
	NumExpertsPerTok             int32    `json:"num_experts_per_tok"`
	MoEIntermediateSize          int32    `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize int32    `json:"shared_expert_intermediate_size"`
	NormTopKProb                 bool     `json:"norm_topk_prob"`
	SlidingWindow                int32    `json:"sliding_window"`
	LayerTypes                   []string `json:"layer_types"`
	HiddenAct                    string   `json:"hidden_act"`
	NumAttentionHeads            int32    `json:"num_attention_heads"`
	NumKeyValueHeads             int32    `json:"num_key_value_heads"`
	VocabSize                    int32    `json:"vocab_size"`
	RMSNormEps                   float32  `json:"rms_norm_eps"`
	RopeTheta                    float32  `json:"rope_theta"`
	HeadDim                      int32    `json:"head_dim"`
	HeadDType                    string   `json:"head_dtype"`
	MaxPositionEmbeddings        int32    `json:"max_position_embeddings"`
	TieWordEmbeddings            bool     `json:"tie_word_embeddings"`

	// Quantization parameters (set during load based on model quantization).
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	Scale float32 `json:"-"`
}

// Model is the Kolibri 1 text-only model.
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config
}

// Layer is a single Kolibri 1 decoder block.
type Layer struct {
	Attention         *Attention
	MLP               *MoE
	PostAttentionNorm *nn.RMSNorm
	PostFFNNorm       *nn.RMSNorm
	AttentionNorm     *nn.RMSNorm
	MLPNorm           *nn.RMSNorm
}

// Attention implements Kolibri 1 attention with Q/K norms.
type Attention struct {
	QProj   nn.LinearLayer
	KProj   nn.LinearLayer
	VProj   nn.LinearLayer
	OProj   nn.LinearLayer
	QNorm   *nn.RMSNorm
	KNorm   *nn.RMSNorm
	Sliding bool
}

// MLP is the feed-forward network with SwiGLU activation.
type MLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

func newModel(root *model.Root) (model.Model, error) {
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

// LoadWeights receives all tensors loaded from the manifest and assigns them
// to model fields.
func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	embedTokens := model.MakeEmbeddingLayer(tensors, "model.embed_tokens", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	if embedTokens == nil {
		return fmt.Errorf("missing embedding weight: model.embed_tokens.weight")
	}
	m.EmbedTokens = embedTokens

	normWeight := tensors["model.norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: model.norm.weight")
	}
	m.Norm = nn.NewRMSNorm(normWeight, m.RMSNormEps)

	if m.TieWordEmbeddings {
		m.LMHead = m.EmbedTokens.AsLinear()
	} else if lmHead := linears.Make("lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else {
		return fmt.Errorf("missing lm_head.weight")
	}

	for i := range m.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("model.layers.%d", i)

		layer := &Layer{
			Attention: &Attention{},
		}

		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.AttentionNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.MLPNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}

		layer.Attention.Sliding = m.LayerTypes[i] == "sliding_attention"
		layer.Attention.QProj = linears.Make(layerPrefix + ".self_attn.q_proj")
		layer.Attention.KProj = linears.Make(layerPrefix + ".self_attn.k_proj")
		layer.Attention.VProj = linears.Make(layerPrefix + ".self_attn.v_proj")
		layer.Attention.OProj = linears.Make(layerPrefix + ".self_attn.o_proj")

		if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
			layer.Attention.QNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.k_norm.weight"]; w != nil {
			layer.Attention.KNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}

		for name, target := range map[string]**nn.RMSNorm{
			".post_attn_norm.weight": &layer.PostAttentionNorm,
			".post_ffn_norm.weight":  &layer.PostFFNNorm,
		} {
			w := tensors[layerPrefix+name]
			if w == nil {
				return fmt.Errorf("layer %d: missing %s", i, name)
			}
			*target = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		var err error
		layer.MLP, err = loadMoE(tensors, linears, layerPrefix, m.Config)
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}

		if layer.AttentionNorm == nil {
			return fmt.Errorf("layer %d: missing input_layernorm", i)
		}
		if layer.MLPNorm == nil {
			return fmt.Errorf("layer %d: missing post_attention_layernorm", i)
		}
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			return fmt.Errorf("layer %d: missing attention projections", i)
		}
		if layer.Attention.QNorm == nil || layer.Attention.KNorm == nil {
			return fmt.Errorf("layer %d: missing attention q/k norms", i)
		}

		m.Layers[i] = layer
	}

	return nil
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
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

	out := m.Norm.Forward(h, m.RMSNormEps)
	return out, out
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	if m.HeadDType == "float32" {
		// Cast before projection so logits are never rounded through BF16,
		// including when the head uses packed quantized weights.
		x = x.AsType(mlx.DTypeFloat32)
	}
	return m.LMHead.Forward(x)
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
		if m.LayerTypes[i] == "sliding_attention" {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	a := l.Attention.Forward(l.AttentionNorm.Forward(x, cfg.RMSNormEps), b, c, positions, B, L, cfg)
	h := mlx.Add(x, l.PostAttentionNorm.Forward(a, cfg.RMSNormEps))
	f := l.MLP.Forward(l.MLPNorm.Forward(h, cfg.RMSNormEps), cfg)
	return mlx.Add(h, l.PostFFNNorm.Forward(f, cfg.RMSNormEps))
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)

	q = a.QNorm.Forward(q, cfg.RMSNormEps)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	if a.Sliding {
		q = mlx.RoPEWithBase(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions)
		k = mlx.RoPEWithBase(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions)
	}

	// MLX SDPA supports grouped-query attention directly (Q heads can be a
	// multiple of K/V heads), so avoid materializing repeated K/V tensors.
	mask := nn.CausalMask()
	var kv nn.SDPAOption
	if c != nil {
		history := c.(cache.Attention).Update(b, k, v)
		kv = nn.WithKVHistory(history)
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
		if a.Sliding {
			mask = mask.Intersect(nn.SlidingWindowMask(b, int(L), int(cfg.SlidingWindow), q.DType()))
		}
	}
	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(mask))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	return m.DownProj.Forward(nn.SwiGLU(m.GateProj, m.UpProj, x))
}

func parseConfig(data []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("kolibri1 config: %w", err)
	}
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 || cfg.HeadDim <= 0 || cfg.HeadDim%2 != 0 || cfg.VocabSize <= 0 || cfg.MaxPositionEmbeddings <= 0 {
		return cfg, fmt.Errorf("kolibri1: invalid model dimensions")
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return cfg, fmt.Errorf("kolibri1: attention heads must be divisible by KV heads")
	}
	if cfg.NumExperts <= 0 || cfg.NumExpertsPerTok <= 0 || cfg.NumExpertsPerTok > cfg.NumExperts || cfg.MoEIntermediateSize <= 0 || cfg.SharedExpertIntermediateSize <= 0 {
		return cfg, fmt.Errorf("kolibri1: invalid expert dimensions")
	}
	if len(cfg.LayerTypes) != int(cfg.NumHiddenLayers) {
		return cfg, fmt.Errorf("kolibri1: layer_types must contain one entry per layer")
	}
	for _, kind := range cfg.LayerTypes {
		switch kind {
		case "sliding_attention":
			if cfg.SlidingWindow <= 0 {
				return cfg, fmt.Errorf("kolibri1: sliding attention needs a positive window")
			}
		case "full_attention":
		default:
			return cfg, fmt.Errorf("kolibri1: unsupported layer type %q", kind)
		}
	}
	if cfg.HiddenAct != "" && cfg.HiddenAct != "silu" {
		return cfg, fmt.Errorf("kolibri1: unsupported activation %q", cfg.HiddenAct)
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	if cfg.RMSNormEps < 0 || cfg.RopeTheta <= 0 {
		return cfg, fmt.Errorf("kolibri1: invalid normalization or RoPE parameters")
	}
	cfg.Scale = float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	return cfg, nil
}
