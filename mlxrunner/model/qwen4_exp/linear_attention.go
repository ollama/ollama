package qwen4_exp

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type linearAttention struct {
	QKV nn.LinearLayer
	Z   nn.LinearLayer
	B   nn.LinearLayer
	A   nn.LinearLayer
	Out nn.LinearLayer

	Conv       *nn.Conv1d
	NormWeight *mlx.Array
	DtBias     *mlx.Array
	ALog       *mlx.Array
	AExp       *mlx.Array
}

func loadLinearAttention(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string) (*linearAttention, error) {
	a := &linearAttention{
		QKV: linears.Make(prefix + ".in_proj_qkv"),
		Z:   linears.Make(prefix + ".in_proj_z"),
		B:   linears.Make(prefix + ".in_proj_b"),
		A:   linears.Make(prefix + ".in_proj_a"),
		Out: linears.Make(prefix + ".out_proj"),
	}
	var err error
	if a.QKV == nil || a.Z == nil || a.B == nil || a.A == nil || a.Out == nil {
		return nil, fmt.Errorf("missing linear attention projection at %s", prefix)
	}
	convWeight, err := requiredArray(tensors, prefix+".conv1d.weight", prefix+".conv1d")
	if err != nil {
		return nil, err
	}
	if convWeight.NumDims() == 3 && convWeight.Dim(1) == 1 {
		convWeight = mlx.Reshape(convWeight, int32(convWeight.Dim(0)), int32(convWeight.Dim(2)))
	}
	if convWeight.NumDims() != 2 {
		return nil, fmt.Errorf("%s.conv1d weight must reduce to 2D, got %dD", prefix, convWeight.NumDims())
	}
	a.Conv = nn.NewConv1d(mlx.ExpandDims(convWeight, 2), nil, 1, 0, 1, int32(convWeight.Dim(0)))
	if a.NormWeight, err = requiredArray(tensors, prefix+".norm.weight", prefix+".norm"); err != nil {
		return nil, err
	}
	if a.DtBias, err = requiredArray(tensors, prefix+".dt_bias", prefix+".dt_proj"); err != nil {
		return nil, err
	}
	if a.ALog, err = requiredArray(tensors, prefix+".A_log", prefix+".a_log"); err != nil {
		return nil, err
	}
	a.AExp = mlx.Exp(a.ALog.AsType(mlx.DTypeFloat32))
	return a, nil
}

var linearAttentionOutputGate = mlx.Compile2(
	"Qwen4LinearAttentionOutputGate",
	func(out, gate *mlx.Array) *mlx.Array {
		dtype := out.DType()
		return mlx.Mul(out.AsType(mlx.DTypeFloat32), mlx.Sigmoid(gate.AsType(mlx.DTypeFloat32))).AsType(dtype)
	},
	mlx.Shapeless(),
)

func (a *linearAttention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	valueDim := cfg.LinearNumValueHeads * cfg.LinearValueHeadDim

	qkv := a.QKV.Forward(x)
	z := mlx.Reshape(a.Z.Forward(x), B, L, cfg.LinearNumValueHeads, cfg.LinearValueHeadDim)
	ba := mlx.Concatenate([]*mlx.Array{a.B.Forward(x), a.A.Forward(x)}, -1)

	convTail := cfg.LinearConvKernelDim - 1
	options := []nn.RecurrentOption{nn.WithConvSiLU()}
	var recurrent *cache.RecurrentCache
	if value, ok := c.(*cache.RecurrentCache); ok {
		recurrent = value
		options = append(options, nn.WithRecurrentHistory(recurrent.Get(b, x.DType())))
		if splits := recurrent.SnapshotSplits(int(L)); len(splits) > 0 {
			options = append(options, nn.WithSnapshotSplits(splits))
		}
	} else {
		options = append(options, nn.WithRecurrentState(
			mlx.Zeros(x.DType(), int(B), int(convTail), qkv.Dim(2)),
			mlx.Zeros(mlx.DTypeFloat32, int(B), int(cfg.LinearNumValueHeads), int(cfg.LinearValueHeadDim), int(cfg.LinearKeyHeadDim)),
		))
	}

	convOut, convState := nn.CausalConv1D(b, qkv, a.Conv, int(convTail), options...)
	out, deltaState := nn.GatedDelta(b, convOut, ba, a.DtBias, a.AExp, options...)
	outType := out.DType()
	out = mlx.RMSNormFn(out, a.NormWeight, cfg.RMSNormEps)
	// Qwen 4 uses a sigmoid Gated DeltaNet output gate rather than the SiLU
	// gate used by Qwen3.5.
	out = linearAttentionOutputGate(out.AsType(outType), z)
	out = a.Out.Forward(mlx.Reshape(out, B, L, valueDim))
	if recurrent != nil {
		recurrent.Put(b, convState, deltaState)
	}
	return out
}
