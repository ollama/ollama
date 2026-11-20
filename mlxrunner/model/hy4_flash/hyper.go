package hy4_flash

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type hyperConnection struct {
	Projection nn.LinearLayer
	Bias       *mlx.Array
	Scale      *mlx.Array
}

func loadHyperConnection(w weights, prefix string) hyperConnection {
	return hyperConnection{
		Projection: nn.NewLinear(w.tensor(prefix+"fn").AsType(mlx.DTypeFloat32), nil),
		Bias:       w.tensor(prefix + "base").AsType(mlx.DTypeFloat32),
		Scale:      w.tensor(prefix + "scale").AsType(mlx.DTypeFloat32).Reshape(-1),
	}
}

func (h hyperConnection) mix(x *mlx.Array, c *textConfig) *mlx.Array {
	flat := x.Reshape(x.Dim(0), x.Dim(1), c.Streams*c.HiddenSize).AsType(mlx.DTypeFloat32)
	rmsInv := mlx.AddScalar(mlx.Mean(mlx.Mul(flat, flat), -1, true), c.RMSEps).Sqrt()
	return mlx.Div(h.Projection.Forward(flat), rmsInv)
}

func (h hyperConnection) prepare(x *mlx.Array, c *textConfig) (input, post *mlx.Array) {
	mix := h.mix(x, c)
	preMix := mix.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.Streams))
	postMix := mix.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(c.Streams, 2*c.Streams))
	pre := mlx.Add(mlx.Mul(preMix, h.Scale.Slice(mlx.Slice(0, 1))), h.Bias.Slice(mlx.Slice(0, c.Streams))).Sigmoid()
	post = mlx.Add(mlx.Mul(postMix, h.Scale.Slice(mlx.Slice(1, 2))), h.Bias.Slice(mlx.Slice(c.Streams, 2*c.Streams))).Sigmoid()
	return mixStreams(x, mlx.AddScalar(pre, c.HCEps)), mlx.AddScalar(mlx.MulScalar(post, c.HCMagnitude), c.HCEps)
}

func (h hyperConnection) reduce(x *mlx.Array, c *textConfig) *mlx.Array {
	pre := mlx.Add(mlx.Mul(h.mix(x, c), h.Scale), h.Bias).Sigmoid()
	return mixStreams(x, mlx.AddScalar(pre, c.HCEps))
}

var mixStreams = mlx.Compile2("hy4_mix_streams", func(x, pre *mlx.Array) *mlx.Array {
	return mlx.Sum(mlx.Mul(x.AsType(mlx.DTypeFloat32), pre.ExpandDims(-1)), 2, false).AsType(x.DType())
}, mlx.Shapeless())

func injectStream(x, branch, post *mlx.Array) *mlx.Array {
	delta := mlx.Mul(branch.AsType(mlx.DTypeFloat32).ExpandDims(2), post.ExpandDims(-1))
	return mlx.Add(x.AsType(mlx.DTypeFloat32), delta).AsType(x.DType())
}
