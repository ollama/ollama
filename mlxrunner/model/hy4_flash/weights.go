package hy4_flash

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// weights reports the first missing required tensor after the layer is wired.
// The placeholders only let a loader finish collecting its fields.
type weights struct {
	tensors map[string]*mlx.Array
	linears model.LinearFactory
	eps     float32
	err     *error
}

func (w weights) missing(name string) {
	if *w.err == nil {
		*w.err = fmt.Errorf("missing required tensor %q", name)
	}
}

func (w weights) tensor(name string) *mlx.Array {
	if a := w.tensors[name]; a != nil {
		return a
	}
	w.missing(name)
	return mlx.Zeros(mlx.DTypeFloat32, 1)
}

func (w weights) linear(name string) nn.LinearLayer {
	if layer := w.linears.Make(name); layer != nil {
		return layer
	}
	w.missing(name + ".weight")
	return nil
}

func (w weights) norm(name string) *nn.RMSNorm {
	return nn.NewRMSNorm(w.tensor(name+".weight"), w.eps)
}
