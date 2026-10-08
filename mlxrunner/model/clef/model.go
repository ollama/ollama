// Package clef implements Cloudflare's Qwen3.5 joint-schema decision model.
package clef

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type config struct {
	HiddenSize    int `json:"hidden_size"`
	Width         int `json:"width"`
	RoutingLayers int `json:"routing_layers"`
	Layers        int `json:"layers"`
	Heads         int `json:"heads"`
	Feedforward   int `json:"feedforward"`
}

type Model struct {
	*qwen3_5.Model
	Head            head
	OutputEmbedding nn.EmbeddingLayer
	config          config
}

func init() { model.Register("ClefForDecision", newModel) }

func newModel(root *model.Root) (model.Model, error) {
	var backboneConfig struct {
		ModelType string `json:"model_type"`
	}
	if err := root.Manifest.ReadConfigJSON("config.json", &backboneConfig); err != nil {
		return nil, err
	}
	if backboneConfig.ModelType != "qwen3_5" {
		return nil, fmt.Errorf("unsupported Clef backbone %q", backboneConfig.ModelType)
	}
	var cfg config
	if err := root.Manifest.ReadConfigJSON("joint_head_config.json", &cfg); err != nil {
		return nil, err
	}
	if cfg.HiddenSize <= 0 || cfg.Width <= 0 || cfg.Heads <= 0 || cfg.Width%cfg.Heads != 0 || cfg.Feedforward <= 0 || cfg.Layers < 1 || cfg.Layers > 16 || cfg.RoutingLayers < 1 || cfg.RoutingLayers > 16 {
		return nil, fmt.Errorf("invalid Clef joint head configuration")
	}
	backbone, err := qwen3_5.NewModel(root)
	if err != nil {
		return nil, err
	}
	m := &Model{Model: backbone.(*qwen3_5.Model), config: cfg}
	if cfg.HiddenSize != int(m.Config.HiddenSize) {
		return nil, fmt.Errorf("Clef head and backbone hidden sizes differ")
	}
	return m, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := m.Model.LoadWeights(tensors); err != nil {
		return err
	}
	c := m.Config
	m.OutputEmbedding = model.MakeEmbeddingLayer(tensors, "lm_head", c.QuantGroupSize, c.QuantBits, c.QuantMode, c.TensorQuant)
	if m.OutputEmbedding == nil {
		return fmt.Errorf("Clef requires its untied output embedding")
	}
	return m.Head.load(tensors, m.config)
}
