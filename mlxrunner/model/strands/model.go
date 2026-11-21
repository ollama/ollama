// Package strands implements Strands Decider's Qwen3.5 pointer-head model.
package strands

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
	HeadType          string             `json:"head_type"`
	PointerDim        int                `json:"pointer_dim"`
	MaxLength         int                `json:"max_length"`
	Temperature       float32            `json:"temperature"`
	TemperatureByKind map[string]float32 `json:"temperature_by_kind"`
}

type Model struct {
	*qwen3_5.Model
	HeadNorm   *nn.LayerNorm
	Query, Key *nn.Linear
	config     config
}

func init() { model.Register("StrandsDeciderForDecision", newModel) }

func newModel(root *model.Root) (model.Model, error) {
	cfg := config{Temperature: 1}
	name := "strands_decider_config.json"
	if _, ok := root.Manifest.ConfigLayer(name); !ok {
		name = "hobson_config.json"
	}
	if err := root.Manifest.ReadConfigJSON(name, &cfg); err != nil {
		return nil, err
	}
	if cfg.HeadType != "pointer" || cfg.PointerDim <= 0 || cfg.MaxLength <= 0 {
		return nil, fmt.Errorf("unsupported Strands Decider head configuration")
	}
	validTemperature := func(t float32) bool { return t > 0 && !math.IsNaN(float64(t)) && !math.IsInf(float64(t), 0) }
	if !validTemperature(cfg.Temperature) {
		return nil, fmt.Errorf("invalid Strands Decider temperature")
	}
	for kind, t := range cfg.TemperatureByKind {
		if !validTemperature(t) {
			return nil, fmt.Errorf("invalid Strands Decider %s temperature", kind)
		}
	}
	backbone, err := qwen3_5.NewModel(root)
	if err != nil {
		return nil, err
	}
	m := &Model{Model: backbone.(*qwen3_5.Model), config: cfg}
	if m.Config.ModelType != "qwen3_5_text" && m.Config.ModelType != "qwen3_5" {
		return nil, fmt.Errorf("unsupported Strands Decider backbone %q", m.Config.ModelType)
	}
	return m, nil
}

func (m *Model) MaxContextLength() int { return min(m.config.MaxLength, m.Model.MaxContextLength()) }

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := m.Model.LoadWeights(tensors); err != nil {
		return err
	}
	hidden, dim := int(m.Config.HiddenSize), m.config.PointerDim
	for name, shape := range map[string][]int{
		"norm.weight": {hidden}, "norm.bias": {hidden},
		"q.weight": {dim, hidden}, "q.bias": {dim},
		"k.weight": {dim, hidden}, "k.bias": {dim},
	} {
		w := tensors[name]
		if w == nil || !slices.Equal(w.Dims(), shape) || w.DType() != mlx.DTypeFloat32 {
			return fmt.Errorf("Strands Decider head requires FP32 %s with shape %v", name, shape)
		}
	}
	m.HeadNorm = &nn.LayerNorm{Weight: tensors["norm.weight"], Bias: tensors["norm.bias"], Eps: 1e-5}
	m.Query = &nn.Linear{Weight: tensors["q.weight"], Bias: tensors["q.bias"]}
	m.Key = &nn.Linear{Weight: tensors["k.weight"], Bias: tensors["k.bias"]}
	return nil
}

// The publisher keeps the pointer head in FP32, including its input normalization.
func (m *Model) pointer(query, options *mlx.Array, kind int) *mlx.Array {
	q := m.Query.Forward(m.HeadNorm.Forward(query.AsType(mlx.DTypeFloat32)))
	k := m.Key.Forward(m.HeadNorm.Forward(options.AsType(mlx.DTypeFloat32)))
	temperature := m.config.Temperature
	if t, ok := m.config.TemperatureByKind[[]string{"noul", "choice", "score"}[kind]]; ok {
		temperature = t
	}
	scale := float32(math.Sqrt(float64(m.config.PointerDim))) * max(temperature, 1e-6)
	return mlx.DivScalar(mlx.Matmul(k, q.Transpose(0, 2, 1)).Squeeze(-1), scale)
}
