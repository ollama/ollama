package lora

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Factors holds the transposed A and B matrices in their exported precision.
type Factors struct {
	AT, BT *mlx.Array
	Scale  float32
}

func (c *Checkpoint) Load() (map[string]*Factors, []*mlx.Array, error) {
	file, err := mlx.LoadSafetensorsNative(c.path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Free()
	factors := make(map[string]*Factors, len(c.pairs))
	arrays := make([]*mlx.Array, 0, len(c.pairs)*2)
	for name, p := range c.pairs {
		a, b := file.Get(p.a), file.Get(p.b)
		if p.projection.OutputLimit > 0 {
			b = b.Slice(mlx.Slice(0, p.projection.OutputLimit), mlx.Slice())
		}
		f := &Factors{AT: transpose(a), BT: transpose(b), Scale: c.scale}
		arrays = append(arrays, f.AT, f.BT)
		factors[name] = f
	}
	return factors, arrays, nil
}

func transpose(w *mlx.Array) *mlx.Array {
	if w.NumDims() == 3 {
		if w.Dim(0) > 1 {
			return w.Transpose(0, 2, 1)
		}
		w = w.Squeeze(0)
	}
	return w.Transpose(1, 0)
}

func (f *Factors) Delta(input *mlx.Array) *mlx.Array {
	return mlx.MulScalar(input.AsType(f.AT.DType()).Matmul(f.AT).Matmul(f.BT), f.Scale)
}

func (f *Factors) Add(base, input *mlx.Array) *mlx.Array {
	if f == nil {
		return base
	}
	return base.AsType(mlx.DTypeFloat32).Add(f.Delta(input)).AsType(base.DType())
}

type linear struct {
	nn.LinearLayer
	factors *Factors
}

func AdaptLinear(base nn.LinearLayer, factors *Factors) nn.LinearLayer {
	if factors == nil {
		return base
	}
	return &linear{base, factors}
}

func (l *linear) Forward(x *mlx.Array) *mlx.Array { return l.factors.Add(l.LinearLayer.Forward(x), x) }
