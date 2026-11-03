// Package kev implements the Qwen3.5/Qwen3.8 Kev decision head.
package kev

import (
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type config struct {
	HeadDim     int     `json:"head_dim"`
	Temperature float32 `json:"temperature"`
	MaxState    int     `json:"max_state"`
	MaxRow      int     `json:"max_row"`
}

type Model struct {
	*qwen3_5.Model
	Query, Key *nn.Linear
	config     config
}

func init() { model.Register("KevForDecision", newModel) }

func newModel(root *model.Root) (model.Model, error) {
	var cfg config
	if err := root.Manifest.ReadConfigJSON("kev_config.json", &cfg); err != nil {
		return nil, err
	}
	if cfg.HeadDim < 1 {
		return nil, fmt.Errorf("invalid Kev head_dim %d", cfg.HeadDim)
	}
	if cfg.MaxState < 1 || cfg.MaxState > 65536 {
		return nil, fmt.Errorf("invalid Kev max_state %d", cfg.MaxState)
	}
	if cfg.MaxRow < cfg.MaxState || cfg.MaxRow > 73728 {
		return nil, fmt.Errorf("invalid Kev max_row %d for max_state %d", cfg.MaxRow, cfg.MaxState)
	}
	if cfg.Temperature <= 0 || math.IsNaN(float64(cfg.Temperature)) || math.IsInf(float64(cfg.Temperature), 0) {
		return nil, fmt.Errorf("invalid Kev temperature %g", cfg.Temperature)
	}
	backbone, err := qwen3_5.NewModel(root)
	if err != nil {
		return nil, err
	}
	m := &Model{Model: backbone.(*qwen3_5.Model), config: cfg}
	m.Config.PreciseGatedDeltaDecay = true
	if m.Config.ModelType != "qwen3_5" && m.Config.ModelType != "qwen3_5_text" {
		return nil, fmt.Errorf("unsupported Kev backbone %q", m.Config.ModelType)
	}
	return m, nil
}

func (m *Model) MaxContextLength() int {
	return min(m.config.MaxRow, m.Model.MaxContextLength())
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := m.Model.LoadWeights(tensors); err != nil {
		return err
	}
	hidden, dim := int(m.Config.HiddenSize), m.config.HeadDim
	for name, shape := range map[string][]int{
		"kev_head.q.weight": {dim, hidden},
		"kev_head.q.bias":   {dim},
		"kev_head.k.weight": {dim, hidden},
		"kev_head.k.bias":   {dim},
	} {
		w := tensors[name]
		if w == nil || w.DType() != mlx.DTypeFloat32 || !slices.Equal(w.Dims(), shape) {
			return fmt.Errorf("Kev requires FP32 %s with shape %v", name, shape)
		}
	}
	m.Query = &nn.Linear{Weight: tensors["kev_head.q.weight"], Bias: tensors["kev_head.q.bias"]}
	m.Key = &nn.Linear{Weight: tensors["kev_head.k.weight"], Bias: tensors["kev_head.k.bias"]}
	return nil
}

// The pointer head runs in FP32 and applies the checkpoint's fitted temperature.
func (m *Model) pointer(query, options *mlx.Array) *mlx.Array {
	q := m.Query.Forward(query.AsType(mlx.DTypeFloat32))
	k := m.Key.Forward(options.AsType(mlx.DTypeFloat32))
	scale := float32(math.Sqrt(float64(m.config.HeadDim))) * m.config.Temperature
	return mlx.DivScalar(mlx.Matmul(k, q.Transpose(0, 2, 1)).Squeeze(-1), scale)
}
