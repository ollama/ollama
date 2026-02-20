//go:build mlx

// Package gemma3 provides the Gemma 3 text model implementation for MLX.
package gemma3

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
	base.Register("Gemma3ForCausalLM", newModel)
	base.Register("Gemma3ForConditionalGeneration", newModel)
}

// TextConfig holds configuration for the Gemma 3 text model.
type TextConfig struct {
	HiddenSize            int32    `json:"hidden_size"`
	NumHiddenLayers       int32    `json:"num_hidden_layers"`
	IntermediateSize      int32    `json:"intermediate_size"`
	NumAttentionHeads     int32    `json:"num_attention_heads"`
	NumKeyValueHeads      int32    `json:"num_key_value_heads"`
	HeadDim               int32    `json:"head_dim"`
	VocabSize             int32    `json:"vocab_size"`
	RMSNormEps            float32  `json:"rms_norm_eps"`
	RopeTheta             float32  `json:"rope_theta"`
	RopeLocalBaseFreq     float32  `json:"rope_local_base_freq"`
	MaxPositionEmbeddings int32    `json:"max_position_embeddings"`
	SlidingWindow         int32    `json:"sliding_window"`
	SlidingWindowPattern  int32    `json:"sliding_window_pattern"`
	LayerTypes            []string `json:"layer_types"`
	TieWordEmbeddings     bool     `json:"tie_word_embeddings"`

	// Quantization parameters (set during load based on model quantization).
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	Scale float32 `json:"-"`
}

// Attention implements Gemma 3 attention with Q/K normalization.
type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer

	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm

	// Precomputed (1 + weight) for Gemma-style RMSNorm.
	QNormScaled *mlx.Array
	KNormScaled *mlx.Array
}

// MLP is the feed-forward network with GELU activation.
type MLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

// DecoderLayer is a single transformer block.
type DecoderLayer struct {
	InputNorm    *nn.RMSNorm
	Attention    *Attention
	PostAttnNorm *nn.RMSNorm
	PreFFNorm    *nn.RMSNorm
	MLP          *MLP
	PostFFNorm   *nn.RMSNorm

	// Precomputed (1 + weight) for Gemma-style RMSNorm.
	InputNormScaled    *mlx.Array
	PostAttnNormScaled *mlx.Array
	PreFFNormScaled    *mlx.Array
	PostFFNormScaled   *mlx.Array

	// Layer metadata.
	IsSliding bool
	LayerIdx  int32
}

// Model is the Gemma 3 text-only model.
type Model struct {
	EmbedTokens *nn.Embedding
	Layers      []*DecoderLayer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	// Precomputed (1 + weight) for Gemma-style RMSNorm.
	NormScaled *mlx.Array

	tok *tokenizer.Tokenizer
	*TextConfig

	weightPrefix string
}

func defaultHeads(numLayers int32) (numHeads, numKVHeads int32) {
	switch numLayers {
	case 34:
		return 8, 4
	case 48:
		return 16, 8
	case 62:
		return 32, 16
	default:
		return 8, 4
	}
}

func parseTextConfig(configData []byte) (TextConfig, bool, error) {
	var cfg TextConfig
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return TextConfig{}, false, fmt.Errorf("parse config: %w", err)
	}

	var wrapped struct {
		TextConfig *TextConfig `json:"text_config"`
	}
	if err := json.Unmarshal(configData, &wrapped); err != nil {
		return TextConfig{}, false, fmt.Errorf("parse nested text config: %w", err)
	}

	fromConditional := wrapped.TextConfig != nil
	if fromConditional {
		cfg = *wrapped.TextConfig

		if cfg.HeadDim == 0 {
			cfg.HeadDim = 256
		}
		if cfg.NumAttentionHeads == 0 {
			cfg.NumAttentionHeads, cfg.NumKeyValueHeads = defaultHeads(cfg.NumHiddenLayers)
		}
		if cfg.NumKeyValueHeads == 0 {
			_, cfg.NumKeyValueHeads = defaultHeads(cfg.NumHiddenLayers)
		}
		if cfg.VocabSize == 0 {
			cfg.VocabSize = 262208
		}
		if cfg.SlidingWindowPattern == 0 && len(cfg.LayerTypes) == 0 {
			cfg.SlidingWindowPattern = 6
		}
		if cfg.MaxPositionEmbeddings == 0 {
			cfg.MaxPositionEmbeddings = 131072
		}
	}

	if cfg.HeadDim == 0 {
		cfg.HeadDim = 256
	}
	if cfg.NumAttentionHeads == 0 {
		cfg.NumAttentionHeads, cfg.NumKeyValueHeads = defaultHeads(cfg.NumHiddenLayers)
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = max(1, cfg.NumAttentionHeads/2)
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	if cfg.RopeLocalBaseFreq == 0 {
		cfg.RopeLocalBaseFreq = 10000
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 262208
	}

	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	return cfg, fromConditional, nil
}

func resolveWeightPrefix(tensors map[string]*mlx.Array) string {
	for _, prefix := range []string{"", "language_model."} {
		if tensors[prefix+"model.embed_tokens.weight"] != nil {
			return prefix
		}
	}
	return ""
}

func isLayerSliding(layerIdx int32, cfg *TextConfig) bool {
	if len(cfg.LayerTypes) > 0 && int(layerIdx) < len(cfg.LayerTypes) {
		return cfg.LayerTypes[layerIdx] == "sliding_attention"
	}
	if cfg.SlidingWindowPattern <= 0 {
		return false
	}
	return (layerIdx+1)%cfg.SlidingWindowPattern != 0
}

func precomputeGemmaScaledWeights(m *Model) {
	if m.Norm != nil {
		m.NormScaled = mlx.AddScalar(m.Norm.Weight, 1.0)
	}

	var scaled []*mlx.Array
	if m.NormScaled != nil {
		scaled = append(scaled, m.NormScaled)
	}

	for _, layer := range m.Layers {
		if layer == nil || layer.Attention == nil {
			continue
		}

		if layer.InputNorm != nil {
			layer.InputNormScaled = mlx.AddScalar(layer.InputNorm.Weight, 1.0)
			scaled = append(scaled, layer.InputNormScaled)
		}
		if layer.PostAttnNorm != nil {
			layer.PostAttnNormScaled = mlx.AddScalar(layer.PostAttnNorm.Weight, 1.0)
			scaled = append(scaled, layer.PostAttnNormScaled)
		}
		if layer.PreFFNorm != nil {
			layer.PreFFNormScaled = mlx.AddScalar(layer.PreFFNorm.Weight, 1.0)
			scaled = append(scaled, layer.PreFFNormScaled)
		}
		if layer.PostFFNorm != nil {
			layer.PostFFNormScaled = mlx.AddScalar(layer.PostFFNorm.Weight, 1.0)
			scaled = append(scaled, layer.PostFFNormScaled)
		}

		if layer.Attention.QNorm != nil {
			layer.Attention.QNormScaled = mlx.AddScalar(layer.Attention.QNorm.Weight, 1.0)
			scaled = append(scaled, layer.Attention.QNormScaled)
		}
		if layer.Attention.KNorm != nil {
			layer.Attention.KNormScaled = mlx.AddScalar(layer.Attention.KNorm.Weight, 1.0)
			scaled = append(scaled, layer.Attention.KNormScaled)
		}
	}

	if len(scaled) > 0 {
		mlx.Eval(scaled...)
	}
}

func newModel(root *model.Root) (base.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	cfg, _, err := parseTextConfig(configData)
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
		Layers:     make([]*DecoderLayer, cfg.NumHiddenLayers),
		TextConfig: &cfg,
		tok:        tok,
	}

	for i := range m.Layers {
		m.Layers[i] = &DecoderLayer{
			LayerIdx:  int32(i),
			IsSliding: isLayerSliding(int32(i), m.TextConfig),
		}
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

	if lmHead := linears.Make(prefix + "lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else if lmHead := linears.Make("lm_head"); lmHead != nil {
		m.LMHead = lmHead
	} else {
		// Gemma usually ties output projection to embeddings.
		m.LMHead = nn.NewLinear(embedWeight, nil)
	}

	for i := int32(0); i < m.NumHiddenLayers; i++ {
		layerPrefix := fmt.Sprintf("%smodel.layers.%d", prefix, i)

		layer := &DecoderLayer{
			LayerIdx:  i,
			IsSliding: isLayerSliding(i, m.TextConfig),
			Attention: &Attention{},
			MLP:       &MLP{},
		}

		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.InputNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.PostAttnNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".pre_feedforward_layernorm.weight"]; w != nil {
			layer.PreFFNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_feedforward_layernorm.weight"]; w != nil {
			layer.PostFFNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}

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

		layer.MLP.GateProj = linears.Make(layerPrefix + ".mlp.gate_proj")
		layer.MLP.UpProj = linears.Make(layerPrefix + ".mlp.up_proj")
		layer.MLP.DownProj = linears.Make(layerPrefix + ".mlp.down_proj")

		if layer.InputNorm == nil {
			return fmt.Errorf("layer %d: missing input_layernorm", i)
		}
		if layer.PostAttnNorm == nil {
			return fmt.Errorf("layer %d: missing post_attention_layernorm", i)
		}
		if layer.PreFFNorm == nil {
			return fmt.Errorf("layer %d: missing pre_feedforward_layernorm", i)
		}
		if layer.PostFFNorm == nil {
			return fmt.Errorf("layer %d: missing post_feedforward_layernorm", i)
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

	precomputeGemmaScaledWeights(m)
	if m.NormScaled == nil {
		return fmt.Errorf("missing precomputed final norm weight")
	}
	collected := mlx.Collect(m)
	mlx.Eval(collected...)

	return nil
}

func (m *Model) Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array {
	dims := tokens.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	h := m.EmbedTokens.Forward(tokens)
	h = mlx.MulScalar(h, float32(math.Sqrt(float64(m.HiddenSize))))

	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, c, B, L, m.TextConfig)
	}

	return mlx.RMSNormFn(h, m.NormScaled, m.RMSNormEps)
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

// NewCaches creates cache objects for all layers.
func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i, layer := range m.Layers {
		if m.SlidingWindow > 0 && layer.IsSliding {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

// FormatPrompt applies the Gemma 3 chat template.
func (m *Model) FormatPrompt(prompt string) string {
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}

func (l *DecoderLayer) Forward(x *mlx.Array, c cache.Cache, B, L int32, cfg *TextConfig) *mlx.Array {
	normed := mlx.RMSNormFn(x, l.InputNormScaled, cfg.RMSNormEps)

	attnOut := l.Attention.Forward(normed, c, B, L, l.IsSliding, cfg)
	attnOut = mlx.RMSNormFn(attnOut, l.PostAttnNormScaled, cfg.RMSNormEps)
	h := mlx.Add(x, attnOut)

	normed = mlx.RMSNormFn(h, l.PreFFNormScaled, cfg.RMSNormEps)

	mlpOut := l.MLP.Forward(normed)
	mlpOut = mlx.RMSNormFn(mlpOut, l.PostFFNormScaled, cfg.RMSNormEps)

	return mlx.Add(h, mlpOut)
}

func (a *Attention) Forward(x *mlx.Array, c cache.Cache, B, L int32, isSliding bool, cfg *TextConfig) *mlx.Array {
	q := a.QProj.Forward(x)
	k := a.KProj.Forward(x)
	v := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)

	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	k = mlx.Transpose(k, 0, 2, 1, 3)

	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	q = mlx.RMSNormFn(q, a.QNormScaled, cfg.RMSNormEps)
	k = mlx.RMSNormFn(k, a.KNormScaled, cfg.RMSNormEps)

	ropeTheta := cfg.RopeTheta
	if isSliding {
		ropeTheta = cfg.RopeLocalBaseFreq
	}

	offset := 0
	if c != nil {
		offset = c.Offset()
	}
	q = mlx.RoPEWithBase(q, int(cfg.HeadDim), false, ropeTheta, 1.0, offset)
	k = mlx.RoPEWithBase(k, int(cfg.HeadDim), false, ropeTheta, 1.0, offset)

	if c != nil {
		k, v = c.Update(k, v)
	}

	repeatFactor := cfg.NumAttentionHeads / cfg.NumKeyValueHeads
	if repeatFactor > 1 {
		k = nn.RepeatKV(k, repeatFactor)
		v = nn.RepeatKV(v, repeatFactor)
	}

	out := mlx.ScaledDotProductAttentionCausal(q, k, v, cfg.Scale, L > 1)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	gate := mlx.GELUApprox(m.GateProj.Forward(x))
	up := m.UpProj.Forward(x)
	return m.DownProj.Forward(mlx.Mul(gate, up))
}
