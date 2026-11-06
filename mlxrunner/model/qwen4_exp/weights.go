package qwen4_exp

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func requiredArray(tensors map[string]*mlx.Array, names ...string) (*mlx.Array, error) {
	for _, name := range names {
		if value := tensors[name]; value != nil {
			return value, nil
		}
	}
	return nil, fmt.Errorf("missing tensor %q", names[0])
}

func requiredLinear(linears model.LinearFactory, name string) (nn.LinearLayer, error) {
	value := linears.Make(name)
	if value == nil {
		return nil, fmt.Errorf("missing linear %q", name)
	}
	return value, nil
}

func loadNorm(tensors map[string]*mlx.Array, name string, eps float32) (*nn.RMSNorm, error) {
	weight, err := requiredArray(tensors, name+".weight", name)
	if err != nil {
		return nil, err
	}
	// Qwen 4 RMSNorm weights are stored zero-centered.
	return nn.NewRMSNorm(mlx.AddScalar(weight, 1), eps), nil
}
