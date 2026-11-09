package nn

import "github.com/ollama/ollama/mlx"

// MultiLinear performs per-head linear projections.
// Weight shape: [num_heads, output_dims, input_dims]
type MultiLinear struct {
	Weight *mlx.Array
}

func NewMultiLinear(weight *mlx.Array) *MultiLinear {
	return &MultiLinear{Weight: weight}
}

func (ml *MultiLinear) Forward(x *mlx.Array) *mlx.Array {
	wT := ml.Weight.Transpose(0, 2, 1)
	return x.Matmul(wT)
}
