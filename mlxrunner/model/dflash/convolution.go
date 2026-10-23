package dflash

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type convolution struct {
	Base       *mlx.Array
	Projection nn.LinearLayer
	groupSize  int
}

func (c *convolution) prepare(x *mlx.Array) (*mlx.Array, *mlx.Array) {
	if c == nil {
		return x, nil
	}
	b, length, width := x.Dim(0), x.Dim(1), x.Dim(2)
	taps := c.Base.Dim(1)
	kernels := c.Projection.Forward(x).Reshape(b, length, 2, taps, width/c.groupSize)
	before := kernels.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, 1), mlx.Slice(), mlx.Slice()).Squeeze(2)
	after := kernels.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(1, 2), mlx.Slice(), mlx.Slice()).Squeeze(2)
	return c.apply(x, before, 0), after
}

func (c *convolution) finish(x, kernel *mlx.Array) *mlx.Array {
	if c == nil {
		return x
	}
	return c.apply(x, kernel, 1)
}

// Convolutions restart at each proposal block; context rows never enter them.
func (c *convolution) apply(x, kernel *mlx.Array, side int) *mlx.Array {
	base := c.Base.Slice(mlx.Slice(side, side+1), mlx.Slice(), mlx.Slice()).Squeeze(0)
	return convolve(x, kernel, base)
}

var convolve = mlx.Compile3("dflash_convolution", convolveEager)

func convolveEager(hidden, dynamic, base *mlx.Array) *mlx.Array {
	b, length, width := hidden.Dim(0), hidden.Dim(1), hidden.Dim(2)
	groups, taps := dynamic.Dim(3), base.Dim(0)
	groupSize := width / groups
	blocks := hidden.Reshape(b, length, groups, groupSize)
	output := mlx.Zeros(hidden.DType(), b, length, groups, groupSize)
	for offset := range taps {
		values := blocks
		if offset > 0 {
			shift := min(offset, length)
			values = mlx.PadConstant(blocks.Slice(mlx.Slice(), mlx.Slice(0, length-shift), mlx.Slice(), mlx.Slice()), []int{1}, []int{shift}, []int{0})
		}
		kernel := base.Slice(mlx.Slice(offset, offset+1), mlx.Slice()).Reshape(1, 1, groups, groupSize).AsType(hidden.DType())
		varying := dynamic.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(offset, offset+1), mlx.Slice()).Reshape(b, length, groups, 1)
		// Keep the BF16 rounding between the static and dynamic contributions.
		output = output.Add(kernel.Multiply(values))
		output = output.Add(varying.Multiply(values))
	}
	return output.Reshape(b, length, width)
}
