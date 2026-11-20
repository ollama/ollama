// Package ripple implements the Ricursive transformer.
package ripple

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/quant"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"

	"github.com/ollama/ollama/mlxrunner/model"
)

func init() { model.Register("RicursiveForCausalLM", newModel) }

// Model retains all loaded arrays; value-residual seeds belong to Forward.
type Model struct {
	*Config
	EmbedTokens       nn.EmbeddingLayer
	Layers            []*Layer
	Norm              *mlx.Array
	LMHead            nn.LinearLayer
	tok               *tokenizer.Tokenizer
	auxHiddenLayers   []int
	draftTensorPrefix string
}

type Layer struct {
	Attention                                             *Attention
	MLP                                                   *MLP
	InputNorm, PostAttentionNorm, PreFFNNorm, PostFFNNorm *mlx.Array
}

type Attention struct {
	QProj, KProj, VProj, OProj, GProj nn.LinearLayer
	QNorm, KNorm, ValueResLambda      *mlx.Array
}

func newModel(root *model.Root) (model.Model, error) {
	cfg, err := loadConfig(root)
	if err != nil {
		return nil, err
	}
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	tc := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	tc.GenerationConfigJSON, _ = root.Manifest.ReadConfig("generation_config.json")
	tc.TokenizerConfigJSON, _ = root.Manifest.ReadConfig("tokenizer_config.json")
	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tc)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}
	m := &Model{Config: cfg, Layers: make([]*Layer, cfg.NumHiddenLayers), tok: tok}
	if root.Draft != nil {
		m.draftTensorPrefix = root.Draft.TensorPrefix
		if m.draftTensorPrefix == "" {
			m.draftTensorPrefix = "draft."
		}
	}
	return m, nil
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer           { return m.tok }
func (m *Model) MaxContextLength() int                     { return int(m.MaxPositionEmbeddings) }
func (m *Model) Unembed(x *mlx.Array) *mlx.Array           { return m.LMHead.Forward(x) }
func (m *Model) NumLayers() int                            { return len(m.Layers) }
func (m *Model) SetAuxHiddenLayers(layers []int)           { m.auxHiddenLayers = slices.Clone(layers) }
func (m *Model) TokenEmbeddings(ids *mlx.Array) *mlx.Array { return m.EmbedTokens.Forward(ids) }
func (m *Model) RawLogits(hidden *mlx.Array) *mlx.Array    { return m.LMHead.Forward(hidden) }

// DraftWindow bounds draft attention to its training context.
func (*Model) DraftWindow() int { return 16384 }

func (m *Model) NewCaches() []cache.Cache {
	out := make([]cache.Cache, len(m.LayerTypes))
	for i, kind := range m.LayerTypes {
		if kind == "sliding_attention" {
			out[i] = cache.NewRotatingKVCache(m.SlidingWindow)
		} else {
			out[i] = cache.NewKVCache()
		}
	}
	return out
}

func (m *Model) weightShapes() map[string][]int {
	h, d, f, v := int(m.HiddenSize), int(m.HeadDim), int(m.IntermediateSize), int(m.VocabSize)
	q, k := int(m.NumAttentionHeads)*d, int(m.NumKeyValueHeads)*d
	shapes := map[string][]int{"model.embed_tokens.weight": {v, h}, "lm_head.weight": {v, h}, "model.norm.weight": {h}}
	for i := range m.NumHiddenLayers {
		p := fmt.Sprintf("model.layers.%d.", i)
		for name, shape := range map[string][]int{
			"input_layernorm.weight": {h}, "post_attention_layernorm.weight": {h}, "pre_feedforward_layernorm.weight": {h}, "post_feedforward_layernorm.weight": {h},
			"self_attn.q_proj.weight": {q, h}, "self_attn.k_proj.weight": {k, h}, "self_attn.v_proj.weight": {k, h}, "self_attn.o_proj.weight": {h, q}, "self_attn.g_proj.weight": {q, h},
			"self_attn.q_norm.weight": {d}, "self_attn.k_norm.weight": {d}, "self_attn.value_res_lambda": {1},
			"mlp.gate_proj.weight": {f, h}, "mlp.up_proj.weight": {f, h}, "mlp.down_proj.weight": {h, f},
		} {
			shapes[p+name] = shape
		}
	}
	return shapes
}

func floatTensor(a *mlx.Array) bool {
	return a.DType() == mlx.DTypeFloat32 || a.DType() == mlx.DTypeBFloat16 || a.DType() == mlx.DTypeFloat16
}

func (m *Model) validateWeights(tensors map[string]*mlx.Array) error {
	allowed := make(map[string]bool)
	for name, shape := range m.weightShapes() {
		w := tensors[name]
		if w == nil {
			return fmt.Errorf("missing weight: %s", name)
		}
		allowed[name] = true
		scales := tensors[name+"_scale"]
		if scales == nil {
			if !floatTensor(w) || !slices.Equal(w.Dims(), shape) {
				return fmt.Errorf("invalid weight %s: dtype %v shape %v, want float %v", name, w.DType(), w.Dims(), shape)
			}
			if m.TensorQuant[name] != nil {
				return fmt.Errorf("missing scale for quantized weight %s", name)
			}
			continue
		}
		if len(shape) != 2 {
			return fmt.Errorf("cannot quantize %s", name)
		}
		if tq := m.TensorQuant[name]; tq != nil && quant.Canonical(tq.QuantType) == "" {
			return fmt.Errorf("unsupported quantization for %s: %q", name, tq.QuantType)
		}
		gs, bits, mode := model.ResolveLinearQuantParams(m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant, name, w, scales)
		if gs <= 0 || (bits != 4 && bits != 8) || (mode != "affine" && mode != "nvfp4" && mode != "mxfp4" && mode != "mxfp8") {
			return fmt.Errorf("invalid quantization for %s", name)
		}
		if (mode == "nvfp4" && (gs != 16 || bits != 4)) || (mode == "mxfp4" && (gs != 32 || bits != 4)) || (mode == "mxfp8" && (gs != 32 || bits != 8)) {
			return fmt.Errorf("invalid quantization group/bits for %s", name)
		}
		packed, ss := []int{shape[0], shape[1] * bits / 32}, []int{shape[0], shape[1] / gs}
		if shape[1]%gs != 0 || shape[1]*bits%32 != 0 || w.DType() != mlx.DTypeUint32 || !slices.Equal(w.Dims(), packed) || !slices.Equal(scales.Dims(), ss) {
			return fmt.Errorf("invalid quantized shape for %s: weight %v scale %v, want %v %v", name, w.Dims(), scales.Dims(), packed, ss)
		}
		allowed[name+"_scale"] = true
		if mode == "affine" {
			qb := tensors[name+"_qbias"]
			if !floatTensor(scales) || qb == nil || !floatTensor(qb) || !slices.Equal(qb.Dims(), ss) {
				return fmt.Errorf("invalid affine scale/bias for %s", name)
			}
			allowed[name+"_qbias"] = true
		} else if scales.DType() != mlx.DTypeUint8 {
			return fmt.Errorf("invalid packed scale dtype for %s", name)
		}
		for _, suffix := range []string{".global_scale", "_scale_2", ".input_scale", ".input_global_scale"} {
			if a := tensors[name+suffix]; a != nil {
				if mode != "nvfp4" || !floatTensor(a) || a.Size() != 1 {
					return fmt.Errorf("invalid %s%s", name, suffix)
				}
				allowed[name+suffix] = true
			}
		}
	}
	for name := range tensors {
		if !allowed[name] && !(m.draftTensorPrefix != "" && strings.HasPrefix(name, m.draftTensorPrefix)) {
			return fmt.Errorf("unexpected weight: %s", name)
		}
	}
	return nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := m.validateWeights(tensors); err != nil {
		return err
	}
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, "model.embed_tokens", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	m.LMHead = linears.Make("lm_head")
	m.Norm = tensors["model.norm.weight"]
	for i := range m.Layers {
		p := fmt.Sprintf("model.layers.%d.", i)
		m.Layers[i] = &Layer{
			InputNorm: tensors[p+"input_layernorm.weight"], PostAttentionNorm: tensors[p+"post_attention_layernorm.weight"], PreFFNNorm: tensors[p+"pre_feedforward_layernorm.weight"], PostFFNNorm: tensors[p+"post_feedforward_layernorm.weight"],
			Attention: &Attention{QProj: linears.Make(p + "self_attn.q_proj"), KProj: linears.Make(p + "self_attn.k_proj"), VProj: linears.Make(p + "self_attn.v_proj"), OProj: linears.Make(p + "self_attn.o_proj"), GProj: linears.Make(p + "self_attn.g_proj"), QNorm: tensors[p+"self_attn.q_norm.weight"], KNorm: tensors[p+"self_attn.k_norm.weight"], ValueResLambda: tensors[p+"self_attn.value_res_lambda"]},
			MLP:       &MLP{GateProj: linears.Make(p + "mlp.gate_proj"), UpProj: linears.Make(p + "mlp.up_proj"), DownProj: linears.Make(p + "mlp.down_proj")},
		}
	}
	return nil
}

type Config struct {
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
	SlidingWindow         int             `json:"sliding_window"`
	LayerTypes            []string        `json:"layer_types"`
	HiddenAct             string          `json:"hidden_act"`
	UseGatedAttention     bool            `json:"use_gated_attention"`
	GateFullPrecision     bool            `json:"gate_full_precision"`
	UseValRes             bool            `json:"use_valres"`
	TieWordEmbeddings     bool            `json:"tie_word_embeddings"`
	AttentionBias         bool            `json:"attention_bias"`
	MLPBias               bool            `json:"mlp_bias"`
	FFNClampThreshold     *float32        `json:"ffn_clamp_threshold"`
	EmbedScale            *float64        `json:"embed_scale"`
	RopeScaling           json.RawMessage `json:"rope_scaling"`
	RopeParameters        json.RawMessage `json:"rope_parameters"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`
	Scale          float32                           `json:"-"`
}

func loadConfig(root *model.Root) (*Config, error) {
	data, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	cfg := &Config{HiddenAct: "silu", RMSNormEps: 1e-6, RopeTheta: 10000, SlidingWindow: 1024, UseGatedAttention: true, GateFullPrecision: true}
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	for _, v := range []struct {
		name  string
		value int32
	}{
		{"hidden_size", cfg.HiddenSize},
		{"intermediate_size", cfg.IntermediateSize},
		{"num_hidden_layers", cfg.NumHiddenLayers},
		{"num_attention_heads", cfg.NumAttentionHeads},
		{"num_key_value_heads", cfg.NumKeyValueHeads},
		{"head_dim", cfg.HeadDim},
		{"vocab_size", cfg.VocabSize},
		{"max_position_embeddings", cfg.MaxPositionEmbeddings},
	} {
		if v.value <= 0 {
			return nil, fmt.Errorf("invalid %s: %d", v.name, v.value)
		}
	}
	if cfg.HeadDim%2 != 0 {
		return nil, fmt.Errorf("head_dim must be even")
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return nil, fmt.Errorf("num_attention_heads must be divisible by num_key_value_heads")
	}
	if int64(cfg.NumAttentionHeads)*int64(cfg.HeadDim) > math.MaxInt32 {
		return nil, fmt.Errorf("num_attention_heads * head_dim overflows")
	}
	if cfg.RMSNormEps <= 0 {
		return nil, fmt.Errorf("rms_norm_eps must be positive")
	}
	if cfg.RopeTheta <= 0 {
		return nil, fmt.Errorf("rope_theta must be positive")
	}
	if cfg.SlidingWindow <= 0 {
		return nil, fmt.Errorf("sliding_window must be positive")
	}
	if len(cfg.LayerTypes) != int(cfg.NumHiddenLayers) {
		return nil, fmt.Errorf("layer_types must have num_hidden_layers entries")
	}
	for i, kind := range cfg.LayerTypes {
		if kind != "sliding_attention" && kind != "full_attention" {
			return nil, fmt.Errorf("unsupported layer_types[%d]: %q", i, kind)
		}
	}
	if cfg.HiddenAct != "silu" {
		return nil, fmt.Errorf("unsupported hidden_act: %q", cfg.HiddenAct)
	}
	for _, v := range []struct {
		name        string
		unsupported bool
	}{
		{"use_gated_attention", !cfg.UseGatedAttention},
		{"gate_full_precision", !cfg.GateFullPrecision},
		{"use_valres", !cfg.UseValRes},
		{"tie_word_embeddings", cfg.TieWordEmbeddings},
		{"attention_bias", cfg.AttentionBias},
		{"mlp_bias", cfg.MLPBias},
		{"rope_scaling", len(cfg.RopeScaling) > 0 && string(cfg.RopeScaling) != "null"},
		{"rope_parameters", len(cfg.RopeParameters) > 0 && string(cfg.RopeParameters) != "null"},
	} {
		if v.unsupported {
			return nil, fmt.Errorf("unsupported %s", v.name)
		}
	}
	if cfg.FFNClampThreshold != nil && *cfg.FFNClampThreshold < 0 {
		return nil, fmt.Errorf("ffn_clamp_threshold must be nonnegative")
	}
	if cfg.EmbedScale != nil && (math.IsInf(*cfg.EmbedScale, 0) || math.IsNaN(*cfg.EmbedScale) || math.Abs(*cfg.EmbedScale) > math.MaxFloat32) {
		return nil, fmt.Errorf("invalid embed_scale")
	}
	cfg.Scale = float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams(root.QuantType())
	if gs := root.GroupSize(); gs > 0 {
		cfg.QuantGroupSize = gs
	}
	cfg.TensorQuant = root.AllTensorQuant()
	return cfg, nil
}
