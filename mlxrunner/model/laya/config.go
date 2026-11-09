// Package laya implements the ModernBERT decision model from
// https://github.com/NandhaKishorM/laya (Apache-2.0).
package laya

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

type encoderConfig struct {
	ModelType        string  `json:"model_type"`
	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	Layers           int     `json:"num_hidden_layers"`
	Heads            int     `json:"num_attention_heads"`
	VocabSize        int     `json:"vocab_size"`
	GlobalEvery      int     `json:"global_attn_every_n_layers"`
	LocalAttention   int     `json:"local_attention"`
	NormEps          float32 `json:"norm_eps"`
	Activation       string  `json:"hidden_activation"`
	AttentionBias    bool    `json:"attention_bias"`
	MLPBias          bool    `json:"mlp_bias"`
	NormBias         bool    `json:"norm_bias"`
	GlobalTheta      float32 `json:"global_rope_theta"`
	LocalTheta       float32 `json:"local_rope_theta"`
	RopeParameters   map[string]struct {
		Theta float32 `json:"rope_theta"`
		Type  string  `json:"rope_type"`
	} `json:"rope_parameters"`
}

type config struct {
	HeadLayers           int                `json:"head_layers"`
	MaxLen               int                `json:"max_len"`
	HeadMaxLen           int                `json:"head_max_len"`
	Temperature          []float32          `json:"temperature"`
	TemperatureByOptions map[string]float32 `json:"temperature_by_options"`
}

func init() { model.Register("LayaForDecision", newModel) }

func newModel(root *model.Root) (model.Model, error) {
	// Laya needs extra precision.
	if err := os.Setenv("MLX_ENABLE_TF32", "0"); err != nil {
		return nil, err
	}

	m := &Model{}
	if err := root.Manifest.ReadConfigJSON("encoder/config.json", &m.encoder); err != nil {
		return nil, err
	}
	if err := root.Manifest.ReadConfigJSON("rl_agent_config.json", &m.config); err != nil {
		return nil, err
	}
	e := &m.encoder
	if e.ModelType != "modernbert" || e.HiddenSize <= 0 || e.Heads <= 0 || e.HiddenSize%e.Heads != 0 || (e.HiddenSize/e.Heads)%2 != 0 || e.Layers <= 0 || e.GlobalEvery <= 0 || e.IntermediateSize <= 0 || e.LocalAttention <= 0 || e.VocabSize <= 0 || e.AttentionBias || e.MLPBias || e.NormBias || (e.Activation != "" && e.Activation != "gelu") {
		return nil, fmt.Errorf("unsupported Laya encoder configuration")
	}
	for _, rope := range e.RopeParameters {
		if rope.Type != "" && rope.Type != "default" {
			return nil, fmt.Errorf("unsupported Laya rotary embedding type %q", rope.Type)
		}
	}
	if m.config.HeadLayers < 0 || m.config.HeadLayers > 8 || m.config.MaxLen <= 0 || m.config.MaxLen > 8192 || m.config.HeadMaxLen < 16 || m.config.HeadMaxLen >= m.config.MaxLen || len(m.config.Temperature) != 3 {
		return nil, fmt.Errorf("invalid Laya decision configuration")
	}
	if e.NormEps == 0 {
		e.NormEps = 1e-5
	}
	if e.GlobalTheta == 0 {
		e.GlobalTheta = 160000
	}
	if e.LocalTheta == 0 {
		e.LocalTheta = 10000
	}
	if v := e.RopeParameters["full_attention"].Theta; v != 0 {
		e.GlobalTheta = v
	}
	if v := e.RopeParameters["sliding_attention"].Theta; v != 0 {
		e.LocalTheta = v
	}
	data, err := root.Manifest.ReadConfig("tokenizer/tokenizer.json")
	if err != nil {
		return nil, err
	}
	m.tok, err = tokenizer.LoadFromBytes(data)
	if err != nil {
		return nil, err
	}
	var tokens struct {
		CLS  string `json:"cls_token"`
		SEP  string `json:"sep_token"`
		Mask string `json:"mask_token"`
	}
	data, err = root.Manifest.ReadConfig("tokenizer/tokenizer_config.json")
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, &tokens); err != nil {
		return nil, err
	}
	for _, token := range []struct {
		name string
		dest *int32
	}{{tokens.CLS, &m.cls}, {tokens.SEP, &m.sep}, {tokens.Mask, &m.mask}} {
		id, ok := m.tok.GetSpecialToken(token.name)
		if !ok || token.name == "" {
			return nil, fmt.Errorf("missing Laya special token %q", token.name)
		}
		*token.dest = id
	}
	m.maskToken = tokens.Mask
	return m, nil
}
