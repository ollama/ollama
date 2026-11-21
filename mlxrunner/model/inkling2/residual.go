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
	s.partial = sumFP32(s.partial, attention, mlx.DTypeFloat32)
	return s
}

// sumFP32 adds two activations in FP32 and rounds the sum to dtype.
func sumFP32(a, b *mlx.Array, dtype mlx.DType) *mlx.Array {
	if dtype == mlx.DTypeBFloat16 {
		return compiledSumBF16(a, b)
	}
	return compiledSumFP32(a, b).AsType(dtype)
}

var compiledSumFP32 = mlx.Compile2("InklingSumFP32", func(a, b *mlx.Array) *mlx.Array {
	return mlx.Add(a.AsType(mlx.DTypeFloat32), b.AsType(mlx.DTypeFloat32))
}, mlx.Shapeless())

var compiledSumBF16 = mlx.Compile2("InklingSumBF16", func(a, b *mlx.Array) *mlx.Array {
	return mlx.Add(a.AsType(mlx.DTypeFloat32), b.AsType(mlx.DTypeFloat32)).AsType(mlx.DTypeBFloat16)
}, mlx.Shapeless())

type messageGate struct {
	Down       *nn.Linear
	GainForget *nn.Linear // FP32 gain rows, then forget rows
}

func (g messageGate) apply(input, message *mlx.Array) *mlx.Array {
	if paired := mlx.RMSNormPairRoundBF16(input, message, residualEpsilon); paired != nil {
		compressed := projectTarget(g.Down, paired).AsType(mlx.DTypeFloat32)
		if mixed := mlx.LoopControllerMix(input, message, compressed, g.GainForget.Weight); mixed != nil {
			return mixed
		}
	}
	return g.applyGraph(input, message)
}

// applyGraph is apply without the Metal kernels. Both gates stay in FP32 up
// to the following layer's input normalization.
func (g messageGate) applyGraph(input, message *mlx.Array) *mlx.Array {
	normalized := func(x *mlx.Array) *mlx.Array {
		return mlx.RMSNormFn(x.AsType(mlx.DTypeFloat32), nil, residualEpsilon).AsType(mlx.DTypeBFloat16)
	}
	compressed := projectTarget(g.Down, mlx.Concatenate([]*mlx.Array{normalized(input), normalized(message)}, -1))
	expansion := projectTarget(g.GainForget, compressed.AsType(mlx.DTypeFloat32))
	width := input.Dim(-1)
	gain := expansion.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, width)).Tanh()
	keep := mlx.AddScalar(mlx.MulScalar(expansion.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(width, 2*width)).Tanh(), -1), 1)
	return mlx.Add(mlx.Mul(input, keep), mlx.Mul(message, gain))
}
