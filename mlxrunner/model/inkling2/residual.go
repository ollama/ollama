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

// mixNorm is modelRMS of the mixture for a vector query, in one launch where
// the residual mixer can normalize its own output.
func (s residualStream) mixNorm(query, weight *mlx.Array, cfg textConfig) *mlx.Array {
	if cfg.NormalizeResidualValues && query.NumDims() == 1 {
		sources := append(s.blocks[:len(s.blocks):len(s.blocks)], s.partial)
		if y := mlx.BlockResidualMixNorm(sources, query, weight, residualEpsilon, cfg.NormEps); y != nil {
			return y
		}
	}
	return modelRMS(s.mix(query, cfg.NormalizeResidualValues), weight, cfg.NormEps)
}

// closeBlock appends the partial sum to the finished blocks and starts the next
// block with the attention update.
func (s residualStream) closeBlock(attention *mlx.Array) residualStream {
	return residualStream{
		blocks:  append(s.blocks[:len(s.blocks):len(s.blocks)], s.partial),
		partial: attention,
	}
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

// loopControllers stacks a looped layer's message gates for the controller
// kernels: attention query, key and value, then feed-forward.
type loopControllers struct {
	Down  *mlx.Array // [4, rank, 2*width] BF16
	Gates *mlx.Array // [4, rank, 2*width] FP32: gain columns, then forget columns
	// Views of the three attention controllers.
	AttentionDown, AttentionGates *mlx.Array
}

// newLoopControllers takes row-contiguous stacks, which the controller kernels
// read in place.
func newLoopControllers(down, gates *mlx.Array) *loopControllers {
	first := func(x *mlx.Array, n int) *mlx.Array { return x.Slice(mlx.Slice(0, n), mlx.Slice(), mlx.Slice()) }
	return &loopControllers{Down: down, Gates: gates, AttentionDown: first(down, 3), AttentionGates: first(gates, 3)}
}

// gate returns views of controller i.
func (c *loopControllers) gate(i int) messageGate {
	row := func(x *mlx.Array) *mlx.Array { return x.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice()) }
	down := row(c.Down)
	return messageGate{Down: nn.NewLinear(down.Squeeze(0), nil), Stack: down, Gates: row(c.Gates)}
}

type messageGate struct {
	Down  *nn.Linear // [rank, 2*width] BF16
	Stack *mlx.Array // Down's rows as [1, rank, 2*width]
	Gates *mlx.Array // [1, rank, 2*width] FP32: gain columns, then forget columns
}

// apply gates input with message index of messages [..., messages, width].
func (g messageGate) apply(input, messages *mlx.Array, index int) *mlx.Array {
	if input.NumDims() == 3 && input.Dim(1) <= shortTargetRows {
		// Short forwards normalize inside the down product.
		if compressed := mlx.LoopControllerDown(input, 1, messages, index, 1, g.Stack, residualEpsilon); compressed != nil {
			if out := mlx.LoopControllerGate(input, 1, messages, index, compressed, g.Gates); out != nil {
				return out.Reshape(input.Dims()...)
			}
		}
	}
	message := selectMessage(messages, index)
	if paired := mlx.RMSNormPairRoundBF16(input, message, residualEpsilon); paired != nil {
		compressed := projectTarget(g.Down, paired).AsType(mlx.DTypeFloat32)
		rank := compressed.Dim(-1)
		compressed = compressed.Reshape(1, compressed.Size()/rank, rank)
		if out := mlx.LoopControllerGate(input, 1, messages, index, compressed, g.Gates); out != nil {
			return out.Reshape(input.Dims()...)
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
	gainForget := nn.NewLinear(g.Gates.Squeeze(0).Transpose(1, 0), nil)
	expansion := projectTarget(gainForget, compressed.AsType(mlx.DTypeFloat32))
	width := input.Dim(-1)
	gain := expansion.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, width)).Tanh()
	keep := mlx.AddScalar(mlx.MulScalar(expansion.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(width, 2*width)).Tanh(), -1), 1)
	return mlx.Add(mlx.Mul(input, keep), mlx.Mul(message, gain))
}
