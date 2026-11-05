// Package flint implements the Flint text model on MLX.
package flint

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
	model.Register("FlintForCausalLM", NewModel)
}

var _ model.Model = (*Model)(nil)

type Config struct {
	HiddenSize                   int32    `json:"hidden_size"`
	IntermediateSize             int32    `json:"intermediate_size"`
	MoeIntermediateSize          int32    `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize int32    `json:"shared_expert_intermediate_size"`
	NumHiddenLayers              int32    `json:"num_hidden_layers"`
	NumAttentionHeads            int32    `json:"num_attention_heads"`
	NumKeyValueHeads             int32    `json:"num_key_value_heads"`
	HeadDim                      int32    `json:"head_dim"`
	VocabSize                    int32    `json:"vocab_size"`
	MaxPositionEmbeddings        int32    `json:"max_position_embeddings"`
	RMSNormEps                   float32  `json:"rms_norm_eps"`
	RopeTheta                    float32  `json:"rope_theta"`
	SlidingWindow                int32    `json:"sliding_window"`
	LayerTypes                   []string `json:"layer_types"`
	NoRopeLayerIndices           []int32  `json:"no_rope_layer_indices"`
	MLPOnlyLayers                []int32  `json:"mlp_only_layers"`
	NumExperts                   int32    `json:"num_experts"`
	NumExpertsPerTok             int32    `json:"num_experts_per_tok"`
	NormTopKProb                 bool     `json:"norm_topk_prob"`
	RouterScoreFunction          string   `json:"router_score_function"`
	SoftmaxType                  string   `json:"softmax_type"`
	TieWordEmbeddings            bool     `json:"tie_word_embeddings"`

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

	// The publisher evaluates the untied output head in float32.
	HeadWeight *mlx.Array
	tok        *tokenizer.Tokenizer
	*Config
}

type Layer struct {
	InputNorm         *nn.RMSNorm
	PostAttentionNorm *nn.RMSNorm
	Attention         *Attention
	MLP               MLPBlock
	Sliding           bool
	UseRoPE           bool
}

type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm
	Sinks *mlx.Array
}

func parseConfig(data []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse Flint config: %w", err)
	}
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.VocabSize <= 0 || cfg.MaxPositionEmbeddings <= 0 {
		return Config{}, fmt.Errorf("invalid Flint model dimensions: hidden=%d layers=%d vocab=%d context=%d", cfg.HiddenSize, cfg.NumHiddenLayers, cfg.VocabSize, cfg.MaxPositionEmbeddings)
	}
	if cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 || cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 || cfg.HeadDim <= 0 {
		return Config{}, fmt.Errorf("invalid Flint attention dimensions: heads=%d kv_heads=%d head_dim=%d", cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim)
	}
	if len(cfg.LayerTypes) != int(cfg.NumHiddenLayers) {
		return Config{}, fmt.Errorf("Flint layer_types has %d entries, want %d", len(cfg.LayerTypes), cfg.NumHiddenLayers)
	}
	for i, layerType := range cfg.LayerTypes {
		if layerType != "full_attention" && layerType != "sliding_attention" {
			return Config{}, fmt.Errorf("layer %d has unsupported attention type %q", i, layerType)
		}
	}
	if cfg.SlidingWindow <= 0 || cfg.RMSNormEps <= 0 || cfg.RopeTheta <= 0 {
		return Config{}, fmt.Errorf("invalid Flint attention settings: window=%d norm_eps=%g rope_theta=%g", cfg.SlidingWindow, cfg.RMSNormEps, cfg.RopeTheta)
	}
	if cfg.NumExperts <= 0 || cfg.NumExpertsPerTok <= 0 || cfg.NumExpertsPerTok > cfg.NumExperts || cfg.MoeIntermediateSize <= 0 {
		return Config{}, fmt.Errorf("invalid Flint MoE dimensions: experts=%d top_k=%d intermediate=%d", cfg.NumExperts, cfg.NumExpertsPerTok, cfg.MoeIntermediateSize)
	}
	if cfg.IntermediateSize <= 0 || cfg.SharedExpertIntermediateSize <= 0 {
		return Config{}, fmt.Errorf("invalid Flint dense or shared expert size: dense=%d shared=%d", cfg.IntermediateSize, cfg.SharedExpertIntermediateSize)
	}
	if cfg.RouterScoreFunction != "sigmoid" || !cfg.NormTopKProb || cfg.SoftmaxType != "learnable" {
		return Config{}, fmt.Errorf("unsupported Flint routing or attention: router=%q normalize=%t softmax=%q", cfg.RouterScoreFunction, cfg.NormTopKProb, cfg.SoftmaxType)
	}
	for _, index := range append(append([]int32{}, cfg.NoRopeLayerIndices...), cfg.MLPOnlyLayers...) {
		if index < 0 || index >= cfg.NumHiddenLayers {
			return Config{}, fmt.Errorf("Flint layer index %d is outside [0,%d)", index, cfg.NumHiddenLayers)
		}
	}
	noRoPE := make(map[int32]bool, len(cfg.NoRopeLayerIndices))
	for _, index := range cfg.NoRopeLayerIndices {
		noRoPE[index] = true
	}
	for index, layerType := range cfg.LayerTypes {
		if (layerType == "full_attention") != noRoPE[int32(index)] {
			return Config{}, fmt.Errorf("layer %d: Flint requires full attention without RoPE and sliding attention with RoPE", index)
		}
	}
	cfg.Scale = float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	return cfg, nil
}

func NewModel(root *model.Root) (model.Model, error) {
	data, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load Flint config: %w", err)
	}
	cfg, err := parseConfig(data)
	if err != nil {
		return nil, err
	}
	cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams(root.QuantType())
	if groupSize := root.GroupSize(); groupSize > 0 {
		cfg.QuantGroupSize = groupSize
	}
	cfg.TensorQuant = root.AllTensorQuant()

	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load Flint tokenizer: %w", err)
	}
	generation, err := root.Manifest.ReadConfig("generation_config.json")
	if err != nil {
		return nil, fmt.Errorf("load Flint generation config: %w", err)
	}
	tokenizerConfig, err := root.Manifest.ReadConfig("tokenizer_config.json")
	if err != nil {
		return nil, fmt.Errorf("load Flint tokenizer config: %w", err)
	}
	tokConfig := &tokenizer.TokenizerConfig{
		ConfigJSON:           data,
		GenerationConfigJSON: generation,
		TokenizerConfigJSON:  tokenizerConfig,
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse Flint tokenizer: %w", err)
	}

	return &Model{Config: &cfg, Layers: make([]*Layer, cfg.NumHiddenLayers), tok: tok}, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, "model.embed_tokens", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing Flint embedding weight")
	}
	finalNorm := tensors["model.norm.weight"]
	if finalNorm == nil {
		return fmt.Errorf("missing Flint final norm")
	}
	m.Norm = nn.NewRMSNorm(finalNorm, m.RMSNormEps)
	if m.TieWordEmbeddings {
		m.LMHead = m.EmbedTokens.AsLinear()
	} else {
		m.LMHead = linears.Make("lm_head")
		if m.LMHead == nil {
			return fmt.Errorf("missing Flint lm_head")
		}
		if weight := tensors["lm_head.weight"]; weight != nil && tensors["lm_head.weight_scale"] == nil {
			m.HeadWeight = weight
		}
	}

	dense := make(map[int32]bool, len(m.MLPOnlyLayers))
	for _, index := range m.MLPOnlyLayers {
		dense[index] = true
	}
	noRoPE := make(map[int32]bool, len(m.NoRopeLayerIndices))
	for _, index := range m.NoRopeLayerIndices {
		noRoPE[index] = true
	}
	for i := range m.Layers {
		prefix := fmt.Sprintf("model.layers.%d", i)
		inputNorm := tensors[prefix+".input_layernorm.weight"]
		postNorm := tensors[prefix+".post_attention_layernorm.weight"]
		qNorm := tensors[prefix+".self_attn.q_norm.weight"]
		kNorm := tensors[prefix+".self_attn.k_norm.weight"]
		sinks := tensors[prefix+".self_attn.softmax_offset"]
		if inputNorm == nil || postNorm == nil || qNorm == nil || kNorm == nil || sinks == nil {
			return fmt.Errorf("layer %d: missing norm or attention sink", i)
		}
		if sinks.NumDims() != 1 || sinks.Dim(0) != int(m.NumAttentionHeads) {
			return fmt.Errorf("layer %d: attention sinks have shape %v, want [%d]", i, sinks.Dims(), m.NumAttentionHeads)
		}
		a := &Attention{
			QProj: linears.Make(prefix + ".self_attn.q_proj"),
			KProj: linears.Make(prefix + ".self_attn.k_proj"),
			VProj: linears.Make(prefix + ".self_attn.v_proj"),
			OProj: linears.Make(prefix + ".self_attn.o_proj"),
			QNorm: nn.NewRMSNorm(qNorm, m.RMSNormEps),
			KNorm: nn.NewRMSNorm(kNorm, m.RMSNormEps),
			Sinks: sinks,
		}
		if a.QProj == nil || a.KProj == nil || a.VProj == nil || a.OProj == nil {
			return fmt.Errorf("layer %d: missing attention projection", i)
		}
		layer := &Layer{
			InputNorm:         nn.NewRMSNorm(inputNorm, m.RMSNormEps),
			PostAttentionNorm: nn.NewRMSNorm(postNorm, m.RMSNormEps),
			Attention:         a,
			Sliding:           m.LayerTypes[i] == "sliding_attention",
			UseRoPE:           !noRoPE[int32(i)],
		}
		if dense[int32(i)] {
			layer.MLP = loadDenseMLP(linears, prefix+".mlp")
		} else {
			moe, err := loadSparseMoE(linears, tensors, prefix, m.Config)
			if err != nil {
				return fmt.Errorf("layer %d: %w", i, err)
			}
			layer.MLP = moe
		}
		if layer.MLP == nil {
			return fmt.Errorf("layer %d: missing MLP weights", i)
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
		if i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
	}
	h = m.Norm.Forward(h, 0)
	return h, h
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	if m.HeadWeight != nil {
		return mlx.Matmul(x.AsType(mlx.DTypeFloat32), mlx.Transpose(m.HeadWeight.AsType(mlx.DTypeFloat32), 1, 0))
	}
	return m.LMHead.Forward(x)
}

func (m *Model) MaxContextLength() int           { return int(m.MaxPositionEmbeddings) }
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i, layer := range m.Layers {
		if layer.Sliding {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	h := mlx.Add(x, l.Attention.Forward(l.InputNorm.Forward(x, 0), b, c, positions, B, L, l, cfg))
	normed := l.PostAttentionNorm.Forward(h, 0)
	if sparse, ok := l.MLP.(*SparseMoE); ok {
		return sparse.ForwardAdd(normed, h, cfg)
	}
	return mlx.Add(h, l.MLP.Forward(normed, cfg))
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, layer *Layer, cfg *Config) *mlx.Array {
	q := mlx.Reshape(a.QProj.Forward(x), B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	k := mlx.Reshape(a.KProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v := mlx.Reshape(a.VProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	q = mlx.Transpose(a.QNorm.Forward(q, 0), 0, 2, 1, 3)
	k = mlx.Transpose(a.KNorm.Forward(k, 0), 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	if layer.UseRoPE {
		q = mlx.RoPEWithBase(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1, positions)
		k = mlx.RoPEWithBase(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1, positions)
	}
	mask := nn.DeviceCausalMask()
	var kv nn.SDPAOption
	if c != nil {
		kv = nn.WithKVHistory(c.(cache.Attention).Update(b, k, v))
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
		if layer.Sliding {
			mask = mask.Intersect(nn.SlidingWindowMaskDevice(b, int(L), int(cfg.SlidingWindow), k.DType()))
		}
	}
	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(mask), nn.WithSinks(a.Sinks))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}
