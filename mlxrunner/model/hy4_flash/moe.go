package hy4_flash

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type denseMLP struct {
	Gate, Up, Down nn.LinearLayer
}

func loadMLP(w weights, prefix string) denseMLP {
	return denseMLP{
		Gate: w.linear(prefix + ".gate_proj"),
		Up:   w.linear(prefix + ".up_proj"),
		Down: w.linear(prefix + ".down_proj"),
	}
}

func (m denseMLP) forward(x *mlx.Array) *mlx.Array {
	return m.Down.Forward(swiGLU(m.Gate.Forward(x), m.Up.Forward(x)))
}

// The activation rounds to the projection dtype before multiplying by up.
var swiGLU = mlx.Compile2("Hy4SwiGLU", func(gate, up *mlx.Array) *mlx.Array {
	activated := mlx.SiLU(gate.AsType(mlx.DTypeFloat32)).AsType(gate.DType())
	return mlx.Mul(activated, up)
}, mlx.Shapeless())

type latentMoE struct {
	Router, RouteDown, RouteUp nn.LinearLayer
	RouterBias                 *mlx.Array
	GateUp, Down               *model.ExpertProjection
	Shared                     denseMLP
}

func (m latentMoE) forward(x *mlx.Array, c *textConfig) *mlx.Array {
	B, L := x.Dim(0), x.Dim(1)
	scores := m.Router.Forward(x.AsType(mlx.DTypeFloat32)).Sigmoid()
	experts := mlx.TopKIndices(mlx.Add(scores, m.RouterBias), c.ExpertsPerTok)
	experts = experts.TakeAlongAxis(mlx.Argsort(experts, -1), -1)
	routeWeights := scores.TakeAlongAxis(experts, -1)
	if c.NormalizeRoute {
		routeWeights = mlx.Div(routeWeights, mlx.AddScalar(mlx.Sum(routeWeights, -1, true), 1e-20))
	}
	routeWeights = mlx.MulScalar(routeWeights, c.RouteScale)

	latent := m.RouteDown.Forward(x).Reshape(B*L, 1, 1, c.LatentWidth)
	experts = experts.Reshape(B*L, c.ExpertsPerTok)
	packed := m.GateUp.Forward(latent, experts, false)
	gate := packed.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.ExpertWidth))
	up := packed.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(c.ExpertWidth, 2*c.ExpertWidth))
	limit := mlx.FromValue(c.ExpertClamp).AsType(gate.DType())
	if c.ExpertClamp > 0 {
		gate = mlx.Minimum(gate, limit)
		up = mlx.Clip(up, limit.Negative(), limit)
	}
	expertOut := m.Down.Forward(swiGLU(gate, up), experts, false).Reshape(B, L, c.ExpertsPerTok, c.LatentWidth)
	// Match the publisher's grouped expert path by accumulating the
	// FP32-weighted outputs before returning to activation precision.
	weighted := mlx.Mul(expertOut, routeWeights.ExpandDims(-1))
	routed := mlx.Sum(weighted, 2, false).AsType(expertOut.DType())
	out := m.RouteUp.Forward(routed)
	return mlx.Add(out, m.Shared.forward(x))
}
