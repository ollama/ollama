package qwen3_5

import (
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// The runtime keeps one layout for the linear-attention input projections:
// InProjQKVZ rows form [q | k | v | z] blocks and InProjBA rows form
// [beta | alpha], so the causal conv consumes a contiguous qkv slice and
// beta/alpha sit at fixed offsets. Split checkpoints concatenate into that
// layout; native combined checkpoints ship rows interleaved per key head and
// are permuted into it once at load.

func packGatedDeltaProjections(qkv, z, b, a, qkvz, ba nn.LinearLayer, cfg *Config) (packedQKVZ, packedBA nn.LinearLayer, err error) {
	switch {
	case (qkvz != nil || ba != nil) && (qkv != nil || z != nil || b != nil || a != nil):
		return nil, nil, fmt.Errorf("both combined and split linear attention projections present")
	case qkvz != nil && ba != nil:
		packedQKVZ, err = permuteProjectionRows(qkvz, nativeQKVZPerm(cfg))
		if err != nil {
			return nil, nil, err
		}
		packedBA, err = permuteProjectionRows(ba, nativeBAPerm(cfg))
		return packedQKVZ, packedBA, err
	case qkv != nil && z != nil && b != nil && a != nil:
		packedQKVZ, err = concatProjectionPair(qkv, z)
		if err != nil {
			return nil, nil, err
		}
		packedBA, err = concatProjectionPair(b, a)
		return packedQKVZ, packedBA, err
	}
	return nil, nil, fmt.Errorf("missing linear attention projections")
}

// nativeQKVZPerm maps packed row -> native row. Native in_proj_qkvz rows are
// grouped per key head as [q(dk) | k(dk) | v(vPerK*dv) | z(vPerK*dv)].
func nativeQKVZPerm(cfg *Config) []int32 {
	nk, nv := int(cfg.LinearNumKeyHeads), int(cfg.LinearNumValueHeads)
	dk, dv := int(cfg.LinearKeyHeadDim), int(cfg.LinearValueHeadDim)
	vPerK := nv / nk
	group := 2*dk + 2*vPerK*dv
	keyDim, valueDim := nk*dk, nv*dv

	perm := make([]int32, 2*keyDim+2*valueDim)
	for kh := range nk {
		base := kh * group
		for i := range dk {
			perm[kh*dk+i] = int32(base + i)
			perm[keyDim+kh*dk+i] = int32(base + dk + i)
		}
		for j := range vPerK * dv {
			perm[2*keyDim+kh*vPerK*dv+j] = int32(base + 2*dk + j)
			perm[2*keyDim+valueDim+kh*vPerK*dv+j] = int32(base + 2*dk + vPerK*dv + j)
		}
	}
	return perm
}

// nativeBAPerm maps packed row -> native row. Native in_proj_ba rows are
// grouped per key head as [beta(vPerK) | alpha(vPerK)].
func nativeBAPerm(cfg *Config) []int32 {
	nk, nv := int(cfg.LinearNumKeyHeads), int(cfg.LinearNumValueHeads)
	vPerK := nv / nk

	perm := make([]int32, 2*nv)
	for kh := range nk {
		for j := range vPerK {
			perm[kh*vPerK+j] = int32(kh*2*vPerK + j)
			perm[nv+kh*vPerK+j] = int32(kh*2*vPerK + vPerK + j)
		}
	}
	return perm
}

func permuteProjectionRows(layer nn.LinearLayer, perm []int32) (nn.LinearLayer, error) {
	indices := mlx.NewArrayInt32(perm, []int32{int32(len(perm))})
	takeRows := func(a *mlx.Array) *mlx.Array {
		if a == nil {
			return nil
		}
		return mlx.Take(a, indices, 0)
	}

	switch l := layer.(type) {
	case *nn.Linear:
		if l.Weight.Dim(0) != len(perm) {
			return nil, fmt.Errorf("projection has %d rows, want %d", l.Weight.Dim(0), len(perm))
		}
		return nn.NewLinear(takeRows(l.Weight), takeRows(l.Bias)), nil
	case *nn.QuantizedLinear:
		if l.Weight.Dim(0) != len(perm) {
			return nil, fmt.Errorf("projection has %d rows, want %d", l.Weight.Dim(0), len(perm))
		}
		globalScale := l.GlobalScale
		if globalScale != nil && globalScale.Size() > 1 {
			globalScale = takeRows(globalScale)
		}
		return &nn.QuantizedLinear{
			Weight:      takeRows(l.Weight),
			Scales:      takeRows(l.Scales),
			QBiases:     takeRows(l.QBiases),
			Bias:        takeRows(l.Bias),
			GlobalScale: globalScale,
			GroupSize:   l.GroupSize,
			Bits:        l.Bits,
			Mode:        l.Mode,
		}, nil
	}
	return nil, fmt.Errorf("unsupported projection type %T", layer)
}

func concatProjectionPair(hi, lo nn.LinearLayer) (nn.LinearLayer, error) {
	hiQ, hiQuant := hi.(*nn.QuantizedLinear)
	loQ, loQuant := lo.(*nn.QuantizedLinear)
	if hiQuant && loQuant &&
		hiQ.GroupSize == loQ.GroupSize && hiQ.Bits == loQ.Bits && hiQ.Mode == loQ.Mode {
		return concatQuantizedPair(hiQ, loQ)
	}
	if hiQuant || loQuant {
		slog.Warn("dequantizing linear attention projections: pair quantization differs")
	}

	hiD, err := denseProjection(hi)
	if err != nil {
		return nil, err
	}
	loD, err := denseProjection(lo)
	if err != nil {
		return nil, err
	}
	weight := mlx.Concatenate([]*mlx.Array{hiD.Weight, loD.Weight.AsType(hiD.Weight.DType())}, 0)
	return nn.NewLinear(weight, concatOptionalRows(hiD.Bias, loD.Bias, hiD.Weight, loD.Weight)), nil
}

func concatQuantizedPair(hi, lo *nn.QuantizedLinear) (nn.LinearLayer, error) {
	var qbiases *mlx.Array
	if hi.QBiases != nil || lo.QBiases != nil {
		if hi.QBiases == nil || lo.QBiases == nil {
			return nil, fmt.Errorf("quantized projections disagree on qbiases")
		}
		qbiases = mlx.Concatenate([]*mlx.Array{hi.QBiases, lo.QBiases}, 0)
	}
	return &nn.QuantizedLinear{
		Weight:      mlx.Concatenate([]*mlx.Array{hi.Weight, lo.Weight}, 0),
		Scales:      mlx.Concatenate([]*mlx.Array{hi.Scales, lo.Scales}, 0),
		QBiases:     qbiases,
		Bias:        concatOptionalRows(hi.Bias, lo.Bias, hi.Scales, lo.Scales),
		GlobalScale: concatGlobalScales(hi.GlobalScale, lo.GlobalScale, hi.Weight, lo.Weight),
		GroupSize:   hi.GroupSize,
		Bits:        hi.Bits,
		Mode:        hi.Mode,
	}, nil
}

func denseProjection(layer nn.LinearLayer) (*nn.Linear, error) {
	switch l := layer.(type) {
	case *nn.Linear:
		return l, nil
	case *nn.QuantizedLinear:
		weight := mlx.Dequantize(l.Weight, l.Scales, l.QBiases, l.GroupSize, l.Bits, l.Mode, l.GlobalScale)
		return nn.NewLinear(weight, l.Bias), nil
	}
	return nil, fmt.Errorf("unsupported projection type %T", layer)
}

// concatOptionalRows concatenates two per-row vectors where nil means zero,
// sized by the paired weights' row counts.
func concatOptionalRows(hi, lo, hiRows, loRows *mlx.Array) *mlx.Array {
	switch {
	case hi == nil && lo == nil:
		return nil
	case hi == nil:
		hi = mlx.Zeros(lo.DType(), hiRows.Dim(0))
	case lo == nil:
		lo = mlx.Zeros(hi.DType(), loRows.Dim(0))
	}
	return mlx.Concatenate([]*mlx.Array{hi, lo.AsType(hi.DType())}, 0)
}

// concatGlobalScales concatenates per-tensor or per-row global scales into a
// per-row vector, where nil means one.
func concatGlobalScales(hi, lo, hiRows, loRows *mlx.Array) *mlx.Array {
	if hi == nil && lo == nil {
		return nil
	}
	expand := func(scale *mlx.Array, rows int32) *mlx.Array {
		switch {
		case scale == nil:
			return mlx.Tile(mlx.FromValues([]float32{1}, 1), []int32{rows})
		case scale.Size() == 1:
			return mlx.Tile(mlx.Reshape(scale.AsType(mlx.DTypeFloat32), 1), []int32{rows})
		default:
			return scale.AsType(mlx.DTypeFloat32)
		}
	}
	return mlx.Concatenate([]*mlx.Array{
		expand(hi, int32(hiRows.Dim(0))),
		expand(lo, int32(loRows.Dim(0))),
	}, 0)
}
