// Package d1 serves Liquid AI's d1 decision checkpoints, which answer System One
// questions from one non-causal forward pass over an LFM3-VL or LFM3-Omni backbone.
package d1

import (
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/lfm3"
)

func init() {
	model.Register("D1Model", NewModel)
}

// The publisher caps each picture at this many pixels before its processor.
const visionMaxPixels = 1_638_400

type d1Config struct {
	Codes      int    `json:"d1_codes"`
	Attention  string `json:"d1_attention"`
	Numbers    bool   `json:"d1_numbers"`
	SetSize    int    `json:"d1_set_size"`
	MaxPixels  int    `json:"d1_vision_max_pixels"`
	BoxEdges   string `json:"d1_box_edges"`
	HiddenSize int32  `json:"-"`
}

// Model answers System One requests on an LFM3-VL or LFM3-Omni backbone.
type Model struct {
	*lfm3.VLModel
	config      d1Config
	calibration calibration
	render      *renderer
}

func NewModel(root *model.Root) (model.Model, error) {
	data, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load d1 config: %w", err)
	}
	var cfg d1Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse d1 config: %w", err)
	}
	if cfg.BoxEdges == "" {
		cfg.BoxEdges = "median"
	}
	// Only the readout these checkpoints were trained with is implemented.
	if cfg.Codes != 2 || cfg.Attention != "set" || !cfg.Numbers || cfg.SetSize < 2 || cfg.SetSize > 16 ||
		cfg.MaxPixels != visionMaxPixels || cfg.BoxEdges != "median" {
		return nil, fmt.Errorf("unsupported d1 configuration (codes %d, attention %q, numbers %v, set size %d, pixels %d, edges %q)",
			cfg.Codes, cfg.Attention, cfg.Numbers, cfg.SetSize, cfg.MaxPixels, cfg.BoxEdges)
	}
	calibration, err := root.Manifest.ReadConfig("calibration.json")
	if err != nil {
		return nil, fmt.Errorf("load d1 calibration: %w", err)
	}
	cal, err := loadCalibration(calibration)
	if err != nil {
		return nil, err
	}
	backbone, err := lfm3.NewDecisionBackbone(root)
	if err != nil {
		return nil, err
	}
	specials, err := specialTokens(root)
	if err != nil {
		return nil, err
	}
	r, err := newRenderer(backbone.Tokenizer(), specials)
	if err != nil {
		return nil, err
	}
	cfg.HiddenSize = backbone.HiddenSize
	return &Model{VLModel: backbone, config: cfg, calibration: cal, render: r}, nil
}

// specialTokens lists the tokenizer's special added tokens.
func specialTokens(root *model.Root) ([]string, error) {
	data, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load d1 tokenizer: %w", err)
	}
	var tok struct {
		AddedTokens []struct {
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &tok); err != nil {
		return nil, fmt.Errorf("parse d1 tokenizer: %w", err)
	}
	var specials []string
	for _, t := range tok.AddedTokens {
		if t.Special && len(t.Content) > 1 {
			specials = append(specials, t.Content)
		}
	}
	return specials, nil
}

// Every row is a fresh non-causal pass: the state reads the questions packed
// after it, so no prefix state can be reused.
func (m *Model) NewCaches() []cache.Cache { return nil }

func (m *Model) Forward(b *batch.Batch, _ []cache.Cache) (*mlx.Array, *mlx.Array) {
	return m.VLModel.Forward(b, nil)
}
