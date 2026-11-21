package inkling2

import (
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

const residualEpsilon = 1e-6

// mixResiduals attends across residual blocks independently at each token.
// A matrix of queries returns a separate mixture for each input projection.
func mixResiduals(sources []*mlx.Array, query *mlx.Array, normalizeValues bool) *mlx.Array {
	if normalizeValues {
		if mixed := mlx.BlockResidualMix(sources, query, residualEpsilon); mixed != nil {
			return mixed
		}
	}
	values := mlx.Stack(sources, -2).AsType(mlx.DTypeFloat32)
	keys := mlx.RMSNormFn(values, nil, residualEpsilon)
	if normalizeValues {
		values = keys
	}
	single := query.NumDims() == 1
	if len(sources) == 1 && single {
		return values.Squeeze(-2)
	}
	if single {
		query = query.ExpandDims(0)
	}
	query = mlx.MulScalar(query, float32(1/math.Sqrt(float64(keys.Dim(-1)))))
	scores := mlx.Sum(mlx.Mul(keys.ExpandDims(-3), query.ExpandDims(-2)), -1, false)
	weights := mlx.SoftmaxAxis(scores, -1, true).ExpandDims(-1)
	mixed := mlx.Sum(mlx.Mul(values.ExpandDims(-3), weights), -2, false)
	if single {
		return mixed.Squeeze(-2)
	}
	return mixed
}

type residualStream struct {
	blocks  []*mlx.Array
	partial *mlx.Array
}

func (s residualStream) mix(query *mlx.Array, normalizeValues bool) *mlx.Array {
	return mixResiduals(append(s.blocks[:len(s.blocks):len(s.blocks)], s.partial), query, normalizeValues)
}

// The feed-forward input sees the attention update before the residual stream
// advances. Within a block that sum is normalized in FP32 without an intervening
// rounding to the activation dtype.
func (s residualStream) afterAttention(attention *mlx.Array, boundary bool) residualStream {
	if boundary {
		return residualStream{
			blocks:  append(s.blocks[:len(s.blocks):len(s.blocks)], s.partial),
			partial: attention,
		}
	}
	s.partial = mlx.Add(s.partial.AsType(mlx.DTypeFloat32), attention.AsType(mlx.DTypeFloat32))
	return s
}

type messageGate struct {
	Down       *nn.Linear
	Gain       *nn.Linear
	Forget     *nn.Linear
	GainForget *nn.Linear
}

func (g messageGate) apply(input, message *mlx.Array) *mlx.Array {
	normalized := func(x *mlx.Array) *mlx.Array {
		return mlx.RMSNormFn(x.AsType(mlx.DTypeFloat32), nil, residualEpsilon).AsType(mlx.DTypeBFloat16)
	}
	if g.GainForget != nil {
		paired := normalized(mlx.Stack([]*mlx.Array{input, message}, -2)).Flatten(-2, -1)
		compressed := projectTarget(g.Down, paired).AsType(mlx.DTypeFloat32)
		expansion := projectTarget(g.GainForget, compressed)
		width := input.Dim(-1)
		gain := expansion.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, width))
		forget := expansion.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(width, 2*width))
		return compiledControllerMix(input, message, gain, forget)[0]
	}
	compressed := projectTarget(g.Down, mlx.Concatenate([]*mlx.Array{normalized(input), normalized(message)}, -1))
	// The two expansion projections produce FP32 before applying tanh.
	compressed = compressed.AsType(mlx.DTypeFloat32)
	gain := projectTarget(g.Gain, compressed).Tanh()
	forget := projectTarget(g.Forget, compressed).Tanh()
	keep := mlx.AddScalar(mlx.MulScalar(forget, -1), 1)
	return mlx.Add(mlx.Mul(input, keep), mlx.Mul(message, gain))
}

// Both output gates are FP32 up to the following layer's input normalization.
var compiledControllerMix = mlx.Compile("InklingLoopController", func(args ...*mlx.Array) []*mlx.Array {
	input, message, gain, forget := args[0], args[1], args[2], args[3]
	keep := mlx.AddScalar(mlx.MulScalar(forget.Tanh(), -1), 1)
	return []*mlx.Array{mlx.Add(mlx.Mul(input, keep), mlx.Mul(message, gain.Tanh()))}
}, mlx.Shapeless())
