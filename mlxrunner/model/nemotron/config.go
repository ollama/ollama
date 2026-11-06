package nemotron

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

// Config holds the dense Nemotron checkpoint configuration.
type Config struct {
	HiddenAct             string          `json:"hidden_act"`
	PartialRotaryFactor   float32         `json:"partial_rotary_factor"`
	AttentionBias         bool            `json:"attention_bias"`
	MLPBias               bool            `json:"mlp_bias"`
	RopeScaling           json.RawMessage `json:"rope_scaling"`
	HiddenSize            int32           `json:"hidden_size"`
	NumHiddenLayers       int32           `json:"num_hidden_layers"`
	IntermediateSize      int32           `json:"intermediate_size"`
	NumAttentionHeads     int32           `json:"num_attention_heads"`
	NumKeyValueHeads      int32           `json:"num_key_value_heads"`
	VocabSize             int32           `json:"vocab_size"`
	NormEps               float32         `json:"norm_eps"`
	RopeTheta             float32         `json:"rope_theta"`
	MaxPositionEmbeddings int32           `json:"max_position_embeddings"`
	TieWordEmbeddings     bool            `json:"tie_word_embeddings"`

	// Quantization parameters (set during load based on model quantization).
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	HeadDim int32   `json:"head_dim"`
	Scale   float32 `json:"-"`
}

func newModel(root *model.Root) (model.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if cfg.HiddenAct != "relu2" || cfg.AttentionBias || cfg.MLPBias {
		return nil, fmt.Errorf("only bias-free relu2 Luciole checkpoints are supported")
	}
	if len(cfg.RopeScaling) != 0 && string(cfg.RopeScaling) != "null" {
		return nil, fmt.Errorf("rope_scaling is not supported by this implementation")
	}
	if cfg.NumHiddenLayers <= 0 || cfg.IntermediateSize <= 0 || cfg.VocabSize <= 0 {
		return nil, fmt.Errorf("invalid model dimensions")
	}
	if cfg.PartialRotaryFactor <= 0 || cfg.PartialRotaryFactor > 1 {
		return nil, fmt.Errorf("invalid partial_rotary_factor")
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
	if cfg.HeadDim == 0 && cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return nil, fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("invalid head_dim: %d", cfg.HeadDim)
	}
	rotaryDims := float32(cfg.HeadDim) * cfg.PartialRotaryFactor
	if int32(rotaryDims)%2 != 0 || rotaryDims != float32(int32(rotaryDims)) {
		return nil, fmt.Errorf("partial rotary dimensions must be an even integer")
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return nil, fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	if cfg.NormEps == 0 {
		cfg.NormEps = 1e-5
	}
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

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
