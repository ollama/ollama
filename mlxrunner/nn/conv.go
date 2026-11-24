package nn

import "github.com/ollama/ollama/mlx"

// Conv1d applies 1D convolution over NLC input.
type Conv1d struct {
	Weight   *mlx.Array
	Bias     *mlx.Array
	Stride   int32
	Padding  int32
	Dilation int32
	Groups   int32
}

func NewConv1d(weight, bias *mlx.Array, stride, padding, dilation, groups int32) *Conv1d {
	if stride <= 0 {
		stride = 1
	}
	if dilation <= 0 {
		dilation = 1
	}
	if groups <= 0 {
		groups = 1
	}
	return &Conv1d{
		Weight:   weight,
		Bias:     bias,
		Stride:   stride,
		Padding:  padding,
		Dilation: dilation,
		Groups:   groups,
	}
}

func (c *Conv1d) Forward(x *mlx.Array) *mlx.Array {
	return mlx.Conv1d(x, c.Weight, c.Bias, c.Stride, c.Padding, c.Dilation, c.Groups)
}
