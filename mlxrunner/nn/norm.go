package nn

import "github.com/ollama/ollama/mlx"

// RMSNorm represents an RMS normalization layer.
type RMSNorm struct {
	Weight *mlx.Array
	Eps    float32
}

func NewRMSNorm(weight *mlx.Array, eps float32) *RMSNorm {
	return &RMSNorm{Weight: weight, Eps: eps}
}

func (rn *RMSNorm) Forward(x *mlx.Array, eps float32) *mlx.Array {
	if eps == 0 {
		eps = rn.Eps
	}
	return mlx.RMSNormFn(x, rn.Weight, eps)
}

// LayerNorm represents a standard layer normalization layer (with bias).
type LayerNorm struct {
	Weight *mlx.Array
	Bias   *mlx.Array
	Eps    float32
}

func (ln *LayerNorm) Forward(x *mlx.Array) *mlx.Array {
	eps := ln.Eps
	if eps == 0 {
		eps = 1e-5
	}
	return mlx.LayerNormFn(x, ln.Weight, ln.Bias, eps)
}
