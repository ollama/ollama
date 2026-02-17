//go:build mlx

// Package qwen3 provides the Qwen3 text model implementation for MLX.
package qwen3

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("Qwen3ForCausalLM", newModel)
}

// Config holds Qwen3 model configuration.
type Config struct {
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	HeadDim               int32   `json:"head_dim"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`

	// Quantization parameters (set during load based on model quantization).
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	Scale     float32 `json:"-"`
	QKNormEps float32 `json:"-"`
}

// Model is the Qwen3 text-only model.
type Model struct {
	EmbedTokens *nn.Embedding
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	tok *tokenizer.Tokenizer
	*Config

	weightPrefix string
}

// Layer is a single Qwen3 decoder block.
type Layer struct {
	Attention     *Attention
	MLP           *MLP
	AttentionNorm *nn.RMSNorm
	MLPNorm       *nn.RMSNorm
}

// Attention implements Qwen3 attention with Q/K norms.
type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm
}

// MLP is the feed-forward network with SwiGLU activation.
type MLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
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
	if cfg.HeadDim == 0 {
		if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
			return nil, fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
		}
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("invalid head_dim: %d", cfg.HeadDim)
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return nil, fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	cfg.QKNormEps = 1e-6

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
	m.weightPrefix = resolveWeightPrefix(tensors)
	prefix := m.weightPrefix
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	embedWeight := tensors[prefix+"model.embed_tokens.weight"]
	if embedWeight == nil {
		return fmt.Errorf("missing embedding weight: %smodel.embed_tokens.weight", prefix)
	}
	m.EmbedTokens = nn.NewEmbedding(embedWeight)

	normWeight := tensors[prefix+"model.norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: %smodel.norm.weight", prefix)
	}
	m.Norm = nn.NewRMSNorm(normWeight, m.RMSNormEps)

	if m.TieWordEmbeddings {
		m.LMHead = nn.NewLinear(embedWeight, nil)
	} else if lmHead := linears.Make(prefix + "lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else if lmHead := linears.Make("lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else {
		// Qwen3 checkpoints commonly tie output projection to embeddings.
		m.LMHead = nn.NewLinear(embedWeight, nil)
	}

	for i := int32(0); i < m.NumHiddenLayers; i++ {
		layerPrefix := fmt.Sprintf("%smodel.layers.%d", prefix, i)

		layer := &Layer{
			Attention: &Attention{},
			MLP:       &MLP{},
		}

		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.AttentionNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.MLPNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}

		layer.Attention.QProj = linears.Make(layerPrefix + ".self_attn.q_proj")
		layer.Attention.KProj = linears.Make(layerPrefix + ".self_attn.k_proj")
		layer.Attention.VProj = linears.Make(layerPrefix + ".self_attn.v_proj")
		layer.Attention.OProj = linears.Make(layerPrefix + ".self_attn.o_proj")

		if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
			layer.Attention.QNorm = nn.NewRMSNorm(w, m.QKNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.k_norm.weight"]; w != nil {
			layer.Attention.KNorm = nn.NewRMSNorm(w, m.QKNormEps)
		}

		layer.MLP.GateProj = linears.Make(layerPrefix + ".mlp.gate_proj")
		layer.MLP.UpProj = linears.Make(layerPrefix + ".mlp.up_proj")
		layer.MLP.DownProj = linears.Make(layerPrefix + ".mlp.down_proj")

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
		if layer.MLP.GateProj == nil || layer.MLP.UpProj == nil || layer.MLP.DownProj == nil {
			return fmt.Errorf("layer %d: missing mlp projections", i)
		}

		m.Layers[i] = layer
	}

	collected := mlx.Collect(m)
	mlx.Eval(collected...)

	return nil
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

	return m.Norm.Forward(h, m.RMSNormEps)
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
	for i := range caches {
		caches[i] = cache.NewKVCache()
	}
	return caches
}

func (l *Layer) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	h := mlx.Add(x, l.Attention.Forward(l.AttentionNorm.Forward(x, cfg.RMSNormEps), c, B, L, cfg))
	return mlx.Add(h, l.MLP.Forward(l.MLPNorm.Forward(h, cfg.RMSNormEps)))
}

func (a *Attention) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)

	q = a.QNorm.Forward(q, cfg.QKNormEps)
	k = a.KNorm.Forward(k, cfg.QKNormEps)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	q = mlx.RoPEWithBase(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, offset)
	k = mlx.RoPEWithBase(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, offset)

	if c != nil {
		k, v = c.Update(k, v)
	}

	// MLX SDPA supports grouped-query attention directly (Q heads can be a
	// multiple of K/V heads), so avoid materializing repeated K/V tensors.
	out := mlx.ScaledDotProductAttentionCausal(q, k, v, cfg.Scale, L > 1)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	return m.DownProj.Forward(mlx.Mul(mlx.SiLU(m.GateProj.Forward(x)), m.UpProj.Forward(x)))
}
